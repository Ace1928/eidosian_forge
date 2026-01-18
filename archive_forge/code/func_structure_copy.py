from functools import wraps
def structure_copy(structure):
    """
    Returns a copy of the given structure (numpy-array, list, iterable, ..).
    """
    if hasattr(structure, 'copy'):
        return structure.copy()
    return iter_copy(structure)