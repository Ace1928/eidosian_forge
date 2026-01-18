import numpy as np
from .version_requirements import require
@require('matplotlib', '>=3.3')
def polygon_clip(rp, cp, r0, c0, r1, c1):
    """Clip a polygon to the given bounding box.

    Parameters
    ----------
    rp, cp : (K,) ndarray of double
        Row and column coordinates of the polygon.
    (r0, c0), (r1, c1) : double
        Top-left and bottom-right coordinates of the bounding box.

    Returns
    -------
    r_clipped, c_clipped : (L,) ndarray of double
        Coordinates of clipped polygon.

    Notes
    -----
    This makes use of Sutherland-Hodgman clipping as implemented in
    AGG 2.4 and exposed in Matplotlib.

    """
    from matplotlib import path, transforms
    poly = path.Path(np.vstack((rp, cp)).T, closed=True)
    clip_rect = transforms.Bbox([[r0, c0], [r1, c1]])
    poly_clipped = poly.clip_to_bbox(clip_rect).to_polygons()[0]
    return (poly_clipped[:, 0], poly_clipped[:, 1])