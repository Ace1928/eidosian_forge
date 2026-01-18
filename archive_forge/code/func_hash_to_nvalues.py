import itertools
import functools
import importlib.util
def hash_to_nvalues(s, nval, seed=None):
    import hashlib
    if seed is None:
        seed = COLORING_SEED
    m = hashlib.sha256()
    m.update(f'{seed}'.encode())
    m.update(s.encode())
    hsh = m.hexdigest()
    b = len(hsh) // nval
    if b == 0:
        raise ValueError(f"Can't extract {nval} values from hash of length {len(hsh)}")
    return tuple((int(hsh[i * b:(i + 1) * b], 16) / 16 ** b for i in range(nval)))