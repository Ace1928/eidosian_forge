from suds import *
from suds.argparser import parse_args
from suds.bindings.binding import Binding
from suds.sax.element import Element
def bodycontent(self, method, args, kwargs):
    wrapped = method.soap.input.body.wrapped
    if wrapped:
        pts = self.bodypart_types(method)
        root = self.document(pts[0])
    else:
        root = []

    def add_param(param_name, param_type, in_choice_context, value):
        """
            Construct request data for the given input parameter.

            Called by our argument parser for every input parameter, in order.

            A parameter's type is identified by its corresponding XSD schema
            element.

            """
        if in_choice_context and value is None:
            return
        pdef = (param_name, param_type)
        p = self.mkparam(method, pdef, value)
        if p is None:
            return
        if not wrapped:
            ns = param_type.namespace('ns0')
            p.setPrefix(ns[0], ns[1])
        root.append(p)
    parse_args(method.name, self.param_defs(method), args, kwargs, add_param, self.options().extraArgumentErrors)
    return root