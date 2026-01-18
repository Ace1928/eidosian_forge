import math
from ..bindings.view_state import ViewState
from .type_checking import is_pandas_df
def get_n_pct(points, proportion=1):
    """Computes the bounding box of the maximum zoom for the specified list of points

    Parameters
    ----------
    points : list of list of float
        List of (x, y) coordinates
    proportion : float, default 1
        Value between 0 and 1 representing the minimum proportion of data to be captured

    Returns
    -------
    list
        k nearest data points
    """
    if proportion == 1:
        return points
    centroid = geometric_mean(points)
    n_to_keep = math.floor(proportion * len(points))
    return k_nearest_neighbors(points, centroid, n_to_keep)