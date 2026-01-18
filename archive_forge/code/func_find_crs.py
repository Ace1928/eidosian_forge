from holoviews.core import Element
from holoviews.operation.element import contours
from holoviews.operation.stats import bivariate_kde
from .. import element as gv_element
from ..element import _Element
from .projection import ( # noqa (API import)
from .resample import resample_geometry # noqa (API import)
def find_crs(op, element):
    """
    Traverses the supplied object looking for coordinate reference
    systems (crs). If multiple clashing reference systems are found
    it will throw an error.
    """
    crss = [crs for crs in element.traverse(lambda x: x.crs, [_Element]) if crs is not None]
    if not crss:
        return {}
    crs = crss[0]
    if any((crs != ocrs for ocrs in crss[1:])):
        raise ValueError('Cannot %s Elements in different coordinate reference systems.' % type(op).__name__)
    return {'crs': crs}