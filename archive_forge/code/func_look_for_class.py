from unittest import TestCase, SkipTest
from hvplot.util import process_xarray  # noqa
def look_for_class(panel, classname, items=None):
    """
    Descend a panel object and find any instances of the given class
    """
    import panel as pn
    if items is None:
        items = []
    if isinstance(panel, pn.layout.ListPanel):
        for p in panel:
            items = look_for_class(p, classname, items)
    elif isinstance(panel, classname):
        items.append(panel)
    return items