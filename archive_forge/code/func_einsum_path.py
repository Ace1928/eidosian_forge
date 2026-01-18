import itertools
import operator
from numpy.core.multiarray import c_einsum
from numpy.core.numeric import asanyarray, tensordot
from numpy.core.overrides import array_function_dispatch
@array_function_dispatch(_einsum_path_dispatcher, module='numpy')
def einsum_path(*operands, optimize='greedy', einsum_call=False):
    """
    einsum_path(subscripts, *operands, optimize='greedy')

    Evaluates the lowest cost contraction order for an einsum expression by
    considering the creation of intermediate arrays.

    Parameters
    ----------
    subscripts : str
        Specifies the subscripts for summation.
    *operands : list of array_like
        These are the arrays for the operation.
    optimize : {bool, list, tuple, 'greedy', 'optimal'}
        Choose the type of path. If a tuple is provided, the second argument is
        assumed to be the maximum intermediate size created. If only a single
        argument is provided the largest input or output array size is used
        as a maximum intermediate size.

        * if a list is given that starts with ``einsum_path``, uses this as the
          contraction path
        * if False no optimization is taken
        * if True defaults to the 'greedy' algorithm
        * 'optimal' An algorithm that combinatorially explores all possible
          ways of contracting the listed tensors and chooses the least costly
          path. Scales exponentially with the number of terms in the
          contraction.
        * 'greedy' An algorithm that chooses the best pair contraction
          at each step. Effectively, this algorithm searches the largest inner,
          Hadamard, and then outer products at each step. Scales cubically with
          the number of terms in the contraction. Equivalent to the 'optimal'
          path for most contractions.

        Default is 'greedy'.

    Returns
    -------
    path : list of tuples
        A list representation of the einsum path.
    string_repr : str
        A printable representation of the einsum path.

    Notes
    -----
    The resulting path indicates which terms of the input contraction should be
    contracted first, the result of this contraction is then appended to the
    end of the contraction list. This list can then be iterated over until all
    intermediate contractions are complete.

    See Also
    --------
    einsum, linalg.multi_dot

    Examples
    --------

    We can begin with a chain dot example. In this case, it is optimal to
    contract the ``b`` and ``c`` tensors first as represented by the first
    element of the path ``(1, 2)``. The resulting tensor is added to the end
    of the contraction and the remaining contraction ``(0, 1)`` is then
    completed.

    >>> np.random.seed(123)
    >>> a = np.random.rand(2, 2)
    >>> b = np.random.rand(2, 5)
    >>> c = np.random.rand(5, 2)
    >>> path_info = np.einsum_path('ij,jk,kl->il', a, b, c, optimize='greedy')
    >>> print(path_info[0])
    ['einsum_path', (1, 2), (0, 1)]
    >>> print(path_info[1])
      Complete contraction:  ij,jk,kl->il # may vary
             Naive scaling:  4
         Optimized scaling:  3
          Naive FLOP count:  1.600e+02
      Optimized FLOP count:  5.600e+01
       Theoretical speedup:  2.857
      Largest intermediate:  4.000e+00 elements
    -------------------------------------------------------------------------
    scaling                  current                                remaining
    -------------------------------------------------------------------------
       3                   kl,jk->jl                                ij,jl->il
       3                   jl,ij->il                                   il->il


    A more complex index transformation example.

    >>> I = np.random.rand(10, 10, 10, 10)
    >>> C = np.random.rand(10, 10)
    >>> path_info = np.einsum_path('ea,fb,abcd,gc,hd->efgh', C, C, I, C, C,
    ...                            optimize='greedy')

    >>> print(path_info[0])
    ['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)]
    >>> print(path_info[1]) 
      Complete contraction:  ea,fb,abcd,gc,hd->efgh # may vary
             Naive scaling:  8
         Optimized scaling:  5
          Naive FLOP count:  8.000e+08
      Optimized FLOP count:  8.000e+05
       Theoretical speedup:  1000.000
      Largest intermediate:  1.000e+04 elements
    --------------------------------------------------------------------------
    scaling                  current                                remaining
    --------------------------------------------------------------------------
       5               abcd,ea->bcde                      fb,gc,hd,bcde->efgh
       5               bcde,fb->cdef                         gc,hd,cdef->efgh
       5               cdef,gc->defg                            hd,defg->efgh
       5               defg,hd->efgh                               efgh->efgh
    """
    path_type = optimize
    if path_type is True:
        path_type = 'greedy'
    if path_type is None:
        path_type = False
    explicit_einsum_path = False
    memory_limit = None
    if path_type is False or isinstance(path_type, str):
        pass
    elif len(path_type) and path_type[0] == 'einsum_path':
        explicit_einsum_path = True
    elif len(path_type) == 2 and isinstance(path_type[0], str) and isinstance(path_type[1], (int, float)):
        memory_limit = int(path_type[1])
        path_type = path_type[0]
    else:
        raise TypeError('Did not understand the path: %s' % str(path_type))
    einsum_call_arg = einsum_call
    input_subscripts, output_subscript, operands = _parse_einsum_input(operands)
    input_list = input_subscripts.split(',')
    input_sets = [set(x) for x in input_list]
    output_set = set(output_subscript)
    indices = set(input_subscripts.replace(',', ''))
    dimension_dict = {}
    broadcast_indices = [[] for x in range(len(input_list))]
    for tnum, term in enumerate(input_list):
        sh = operands[tnum].shape
        if len(sh) != len(term):
            raise ValueError('Einstein sum subscript %s does not contain the correct number of indices for operand %d.' % (input_subscripts[tnum], tnum))
        for cnum, char in enumerate(term):
            dim = sh[cnum]
            if dim == 1:
                broadcast_indices[tnum].append(char)
            if char in dimension_dict.keys():
                if dimension_dict[char] == 1:
                    dimension_dict[char] = dim
                elif dim not in (1, dimension_dict[char]):
                    raise ValueError("Size of label '%s' for operand %d (%d) does not match previous terms (%d)." % (char, tnum, dimension_dict[char], dim))
            else:
                dimension_dict[char] = dim
    broadcast_indices = [set(x) for x in broadcast_indices]
    size_list = [_compute_size_by_dict(term, dimension_dict) for term in input_list + [output_subscript]]
    max_size = max(size_list)
    if memory_limit is None:
        memory_arg = max_size
    else:
        memory_arg = memory_limit
    inner_product = sum((len(x) for x in input_sets)) - len(indices) > 0
    naive_cost = _flop_count(indices, inner_product, len(input_list), dimension_dict)
    if explicit_einsum_path:
        path = path_type[1:]
    elif path_type is False or len(input_list) in [1, 2] or indices == output_set:
        path = [tuple(range(len(input_list)))]
    elif path_type == 'greedy':
        path = _greedy_path(input_sets, output_set, dimension_dict, memory_arg)
    elif path_type == 'optimal':
        path = _optimal_path(input_sets, output_set, dimension_dict, memory_arg)
    else:
        raise KeyError('Path name %s not found', path_type)
    cost_list, scale_list, size_list, contraction_list = ([], [], [], [])
    for cnum, contract_inds in enumerate(path):
        contract_inds = tuple(sorted(list(contract_inds), reverse=True))
        contract = _find_contraction(contract_inds, input_sets, output_set)
        out_inds, input_sets, idx_removed, idx_contract = contract
        cost = _flop_count(idx_contract, idx_removed, len(contract_inds), dimension_dict)
        cost_list.append(cost)
        scale_list.append(len(idx_contract))
        size_list.append(_compute_size_by_dict(out_inds, dimension_dict))
        bcast = set()
        tmp_inputs = []
        for x in contract_inds:
            tmp_inputs.append(input_list.pop(x))
            bcast |= broadcast_indices.pop(x)
        new_bcast_inds = bcast - idx_removed
        if not len(idx_removed & bcast):
            do_blas = _can_dot(tmp_inputs, out_inds, idx_removed)
        else:
            do_blas = False
        if cnum - len(path) == -1:
            idx_result = output_subscript
        else:
            sort_result = [(dimension_dict[ind], ind) for ind in out_inds]
            idx_result = ''.join([x[1] for x in sorted(sort_result)])
        input_list.append(idx_result)
        broadcast_indices.append(new_bcast_inds)
        einsum_str = ','.join(tmp_inputs) + '->' + idx_result
        contraction = (contract_inds, idx_removed, einsum_str, input_list[:], do_blas)
        contraction_list.append(contraction)
    opt_cost = sum(cost_list) + 1
    if len(input_list) != 1:
        raise RuntimeError('Invalid einsum_path is specified: {} more operands has to be contracted.'.format(len(input_list) - 1))
    if einsum_call_arg:
        return (operands, contraction_list)
    overall_contraction = input_subscripts + '->' + output_subscript
    header = ('scaling', 'current', 'remaining')
    speedup = naive_cost / opt_cost
    max_i = max(size_list)
    path_print = '  Complete contraction:  %s\n' % overall_contraction
    path_print += '         Naive scaling:  %d\n' % len(indices)
    path_print += '     Optimized scaling:  %d\n' % max(scale_list)
    path_print += '      Naive FLOP count:  %.3e\n' % naive_cost
    path_print += '  Optimized FLOP count:  %.3e\n' % opt_cost
    path_print += '   Theoretical speedup:  %3.3f\n' % speedup
    path_print += '  Largest intermediate:  %.3e elements\n' % max_i
    path_print += '-' * 74 + '\n'
    path_print += '%6s %24s %40s\n' % header
    path_print += '-' * 74
    for n, contraction in enumerate(contraction_list):
        inds, idx_rm, einsum_str, remaining, blas = contraction
        remaining_str = ','.join(remaining) + '->' + output_subscript
        path_run = (scale_list[n], einsum_str, remaining_str)
        path_print += '\n%4d    %24s %40s' % path_run
    path = ['einsum_path'] + path
    return (path, path_print)