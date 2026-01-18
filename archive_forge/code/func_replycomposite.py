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
def replycomposite(self, rtypes, nodes):
    """
        Construct a I{composite} reply.

        Called for replies with multiple output nodes.

        @param rtypes: A list of known return I{types}.
        @type rtypes: [L{suds.xsd.sxbase.SchemaObject},...]
        @param nodes: A collection of XML nodes.
        @type nodes: [L{Element},...]
        @return: The I{unmarshalled} composite object.
        @rtype: L{Object},...

        """
    dictionary = {}
    for rt in rtypes:
        dictionary[rt.name] = rt
    unmarshaller = self.unmarshaller()
    composite = Factory.object('reply')
    for node in nodes:
        tag = node.name
        rt = dictionary.get(tag)
        if rt is None:
            if node.get('id') is None and (not self.options().allowUnknownMessageParts):
                message = '<%s/> not mapped to message part' % (tag,)
                raise Exception(message)
            continue
        resolved = rt.resolve(nobuiltin=True)
        sobject = unmarshaller.process(node, resolved)
        value = getattr(composite, tag, None)
        if value is None:
            if rt.multi_occurrence():
                value = []
                setattr(composite, tag, value)
                value.append(sobject)
            else:
                setattr(composite, tag, sobject)
        else:
            if not isinstance(value, list):
                value = [value]
                setattr(composite, tag, value)
            value.append(sobject)
    return composite