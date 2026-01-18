from suds import *
from suds.argparser import parse_args
from suds.bindings.binding import Binding
from suds.sax.element import Element
def replycontent(self, method, body):
    if method.soap.output.body.wrapped:
        return body[0].children
    return body.children