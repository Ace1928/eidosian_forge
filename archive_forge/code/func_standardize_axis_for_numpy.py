import operator
import warnings
def standardize_axis_for_numpy(axis):
    """Standardize an axis to a tuple if it is a list in the numpy backend."""
    return tuple(axis) if isinstance(axis, list) else axis