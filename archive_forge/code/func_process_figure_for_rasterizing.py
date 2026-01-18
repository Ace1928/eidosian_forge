from matplotlib.transforms import Bbox, TransformedBbox, Affine2D
def process_figure_for_rasterizing(fig, bbox_inches_restore, fixed_dpi=None):
    """
    A function that needs to be called when figure dpi changes during the
    drawing (e.g., rasterizing).  It recovers the bbox and re-adjust it with
    the new dpi.
    """
    bbox_inches, restore_bbox = bbox_inches_restore
    restore_bbox()
    r = adjust_bbox(fig, bbox_inches, fixed_dpi)
    return (bbox_inches, r)