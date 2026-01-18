from functools import reduce
from sympy.core.sorting import default_sort_key
from sympy.core.containers import Tuple
from sympy.core.mul import Mul
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.utilities import numbered_symbols
from sympy.physics.quantum.gate import Gate
def convert_to_symbolic_indices(seq, start=None, gen=None, qubit_map=None):
    """Returns the circuit with symbolic indices and the
    dictionary mapping symbolic indices to real indices.

    The mapping is 1 to 1 and onto (bijective).

    Parameters
    ==========

    seq : tuple, Gate/Integer/tuple or Mul
        A tuple of Gate, Integer, or tuple objects, or a Mul
    start : Symbol
        An optional starting symbolic index
    gen : object
        An optional numbered symbol generator
    qubit_map : dict
        An existing mapping of symbolic indices to real indices

    All symbolic indices have the format 'i#', where # is
    some number >= 0.
    """
    if isinstance(seq, Mul):
        seq = seq.args
    index_gen = numbered_symbols(prefix='i', start=-1)
    cur_ndx = next(index_gen)
    ndx_map = {}

    def create_inverse_map(symb_to_real_map):
        rev_items = lambda item: (item[1], item[0])
        return dict(map(rev_items, symb_to_real_map.items()))
    if start is not None:
        if not isinstance(start, Symbol):
            msg = 'Expected Symbol for starting index, got %r.' % start
            raise TypeError(msg)
        cur_ndx = start
    if gen is not None:
        if not isinstance(gen, numbered_symbols().__class__):
            msg = 'Expected a generator, got %r.' % gen
            raise TypeError(msg)
        index_gen = gen
    if qubit_map is not None:
        if not isinstance(qubit_map, dict):
            msg = 'Expected dict for existing map, got ' + '%r.' % qubit_map
            raise TypeError(msg)
        ndx_map = qubit_map
    ndx_map = _sympify_qubit_map(ndx_map)
    inv_map = create_inverse_map(ndx_map)
    sym_seq = ()
    for item in seq:
        if isinstance(item, Gate):
            result = convert_to_symbolic_indices(item.args, qubit_map=ndx_map, start=cur_ndx, gen=index_gen)
            sym_item, new_map, cur_ndx, index_gen = result
            ndx_map.update(new_map)
            inv_map = create_inverse_map(ndx_map)
        elif isinstance(item, (tuple, Tuple)):
            result = convert_to_symbolic_indices(item, qubit_map=ndx_map, start=cur_ndx, gen=index_gen)
            sym_item, new_map, cur_ndx, index_gen = result
            ndx_map.update(new_map)
            inv_map = create_inverse_map(ndx_map)
        elif item in inv_map:
            sym_item = inv_map[item]
        else:
            cur_ndx = next(gen)
            ndx_map[cur_ndx] = item
            inv_map[item] = cur_ndx
            sym_item = cur_ndx
        if isinstance(item, Gate):
            sym_item = item.__class__(*sym_item)
        sym_seq = sym_seq + (sym_item,)
    return (sym_seq, ndx_map, cur_ndx, index_gen)