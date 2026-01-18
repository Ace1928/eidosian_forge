import matplotlib.path as mpath
import numpy as np
def bbox_to_path(bbox):
    """
    Turn the given :class:`matplotlib.transforms.Bbox` instance into
    a :class:`matplotlib.path.Path` instance.

    """
    verts = np.array([[bbox.x0, bbox.y0], [bbox.x1, bbox.y0], [bbox.x1, bbox.y1], [bbox.x0, bbox.y1], [bbox.x0, bbox.y0]])
    return mpath.Path(verts)