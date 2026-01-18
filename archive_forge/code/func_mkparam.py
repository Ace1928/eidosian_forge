from suds import *
from suds.argparser import parse_args
from suds.bindings.binding import Binding
from suds.sax.element import Element
def mkparam(self, method, pdef, object):
    """
        Expand list parameters into individual parameters each with the type
        information. This is because in document arrays are simply
        multi-occurrence elements.

        """
    if isinstance(object, (list, tuple)):
        return [self.mkparam(method, pdef, item) for item in object]
    return super(Document, self).mkparam(method, pdef, object)