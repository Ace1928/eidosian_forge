from suds import *
from suds.argparser import parse_args
from suds.bindings.binding import Binding
from suds.sax.element import Element
def param_defs(self, method):
    """Get parameter definitions for document literal."""
    pts = self.bodypart_types(method)
    if not method.soap.input.body.wrapped:
        return pts
    pt = pts[0][1].resolve()
    return [(c.name, c, a) for c, a in pt if not c.isattr()]