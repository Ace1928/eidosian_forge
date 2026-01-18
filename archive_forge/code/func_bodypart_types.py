from suds import *
from suds.sax import Namespace
from suds.sax.document import Document
from suds.sax.element import Element
from suds.sudsobject import Factory
from suds.mx import Content
from suds.mx.literal import Literal as MxLiteral
from suds.umx.typed import Typed as UmxTyped
from suds.bindings.multiref import MultiRef
from suds.xsd.query import TypeQuery, ElementQuery
from suds.xsd.sxbasic import Element as SchemaElement
from suds.options import Options
from suds.plugin import PluginContainer
from copy import deepcopy
def bodypart_types(self, method, input=True):
    """
        Get a list of I{parameter definitions} (pdefs) defined for the
        specified method.

        An input I{pdef} is a (I{name}, L{xsd.sxbase.SchemaObject}) tuple,
        while an output I{pdef} is a L{xsd.sxbase.SchemaObject}.

        @param method: A service method.
        @type method: I{service.Method}
        @param input: Defines input/output message.
        @type input: boolean
        @return:  A list of parameter definitions
        @rtype: [I{pdef},...]

        """
    if input:
        parts = method.soap.input.body.parts
    else:
        parts = method.soap.output.body.parts
    return [self.__part_type(p, input) for p in parts]