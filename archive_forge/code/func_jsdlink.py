from .widget import Widget, register, widget_serialization
from .widget_core import CoreWidget
from traitlets import Unicode, Tuple, Instance, TraitError
def jsdlink(source, target):
    """Link a source widget attribute with a target widget attribute.

    The link is created in the front-end and does not rely on a roundtrip
    to the backend.

    Parameters
    ----------
    source : a (Widget, 'trait_name') tuple for the source trait
    target : a (Widget, 'trait_name') tuple for the target trait

    Examples
    --------

    >>> c = dlink((src_widget, 'value'), (tgt_widget, 'value'))
    """
    return DirectionalLink(source, target)