import numpy as np
import shapely
def patch_from_polygon(polygon, **kwargs):
    """
    Gets a Matplotlib patch from a (Multi)Polygon.

    Note: this function is experimental, and mainly targetting (interactive)
    exploration, debugging and illustration purposes.

    Parameters
    ----------
    polygon : shapely.Polygon or shapely.MultiPolygon
    **kwargs
        Additional keyword arguments passed to the matplotlib Patch.

    Returns
    -------
    Matplotlib artist (PathPatch)
    """
    from matplotlib.patches import PathPatch
    return PathPatch(_path_from_polygon(polygon), **kwargs)