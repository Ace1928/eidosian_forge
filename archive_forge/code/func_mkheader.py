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
def mkheader(self, method, hdef, object):
    """
        Builds a soapheader for the specified I{method} using the header
        definition (hdef) and the specified value (object).

        @param method: A method name.
        @type method: str
        @param hdef: A header definition.
        @type hdef: tuple: (I{name}, L{xsd.sxbase.SchemaObject})
        @param object: The header value.
        @type object: any
        @return: The parameter fragment.
        @rtype: L{Element}

        """
    marshaller = self.marshaller()
    if isinstance(object, (list, tuple)):
        return [self.mkheader(method, hdef, item) for item in object]
    content = Content(tag=hdef[0], value=object, type=hdef[1])
    return marshaller.process(content)