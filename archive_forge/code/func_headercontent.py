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
def headercontent(self, method):
    """
        Get the content for the SOAP I{Header} node.

        @param method: A service method.
        @type method: I{service.Method}
        @return: The XML content for the <body/>.
        @rtype: [L{Element},...]

        """
    content = []
    wsse = self.options().wsse
    if wsse is not None:
        content.append(wsse.xml())
    headers = self.options().soapheaders
    if not isinstance(headers, (tuple, list, dict)):
        headers = (headers,)
    elif not headers:
        return content
    pts = self.headpart_types(method)
    if isinstance(headers, (tuple, list)):
        n = 0
        for header in headers:
            if isinstance(header, Element):
                content.append(deepcopy(header))
                continue
            if len(pts) == n:
                break
            h = self.mkheader(method, pts[n], header)
            ns = pts[n][1].namespace('ns0')
            h.setPrefix(ns[0], ns[1])
            content.append(h)
            n += 1
    else:
        for pt in pts:
            header = headers.get(pt[0])
            if header is None:
                continue
            h = self.mkheader(method, pt, header)
            ns = pt[1].namespace('ns0')
            h.setPrefix(ns[0], ns[1])
            content.append(h)
    return content