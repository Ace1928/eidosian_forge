from collections import OrderedDict
import numpy as np
from .parser import get_symbol
def rand_equation(n, reg, n_out=0, d_min=2, d_max=9, seed=None, global_dim=False, return_size_dict=False):
    """Generate a random contraction and shapes.

    Parameters
    ----------
    n : int
        Number of array arguments.
    reg : int
        'Regularity' of the contraction graph. This essentially determines how
        many indices each tensor shares with others on average.
    n_out : int, optional
        Number of output indices (i.e. the number of non-contracted indices).
        Defaults to 0, i.e., a contraction resulting in a scalar.
    d_min : int, optional
        Minimum dimension size.
    d_max : int, optional
        Maximum dimension size.
    seed: int, optional
        If not None, seed numpy's random generator with this.
    global_dim : bool, optional
        Add a global, 'broadcast', dimension to every operand.
    return_size_dict : bool, optional
        Return the mapping of indices to sizes.

    Returns
    -------
    eq : str
        The equation string.
    shapes : list[tuple[int]]
        The array shapes.
    size_dict : dict[str, int]
        The dict of index sizes, only returned if ``return_size_dict=True``.

    Examples
    --------
    >>> eq, shapes = rand_equation(n=10, reg=4, n_out=5, seed=42)
    >>> eq
    'oyeqn,tmaq,skpo,vg,hxui,n,fwxmr,hitplcj,kudlgfv,rywjsb->cebda'

    >>> shapes
    [(9, 5, 4, 5, 4),
     (4, 4, 8, 5),
     (9, 4, 6, 9),
     (6, 6),
     (6, 9, 7, 8),
     (4,),
     (9, 3, 9, 4, 9),
     (6, 8, 4, 6, 8, 6, 3),
     (4, 7, 8, 8, 6, 9, 6),
     (9, 5, 3, 3, 9, 5)]
    """
    if seed is not None:
        np.random.seed(seed)
    num_inds = n * reg // 2 + n_out
    inputs = ['' for _ in range(n)]
    output = []
    size_dict = OrderedDict(((get_symbol(i), np.random.randint(d_min, d_max + 1)) for i in range(num_inds)))

    def gen():
        for i, ix in enumerate(size_dict):
            if i < n_out:
                output.append(ix)
                yield ix
            else:
                yield ix
                yield ix
    for i, ix in enumerate(np.random.permutation(list(gen()))):
        if i < n:
            inputs[i] += ix
        else:
            where = np.random.randint(0, n)
            while ix in inputs[where]:
                where = np.random.randint(0, n)
            inputs[where] += ix
    if global_dim:
        gdim = get_symbol(num_inds)
        size_dict[gdim] = np.random.randint(d_min, d_max + 1)
        for i in range(n):
            inputs[i] += gdim
        output += gdim
    output = ''.join(np.random.permutation(output))
    eq = '{}->{}'.format(','.join(inputs), output)
    shapes = [tuple((size_dict[ix] for ix in op)) for op in inputs]
    ret = (eq, shapes)
    if return_size_dict:
        ret += (size_dict,)
    return ret