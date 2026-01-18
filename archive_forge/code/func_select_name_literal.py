import math
import decimal
import operator
from copy import copy
from ..datatypes import AnyURI
from ..exceptions import ElementPathKeyError, ElementPathTypeError
from ..helpers import collapse_white_spaces, node_position
from ..datatypes import AbstractDateTime, Duration, DayTimeDuration, \
from ..xpath_context import XPathSchemaContext
from ..namespaces import XMLNS_NAMESPACE, XSD_NAMESPACE
from ..schema_proxy import AbstractSchemaProxy
from ..xpath_nodes import XPathNode, ElementNode, AttributeNode, DocumentNode
from ..xpath_tokens import XPathToken
from .xpath1_parser import XPath1Parser
@method('(name)')
def select_name_literal(self, context=None):
    if context is None:
        raise self.missing_context()
    elif isinstance(context, XPathSchemaContext):
        yield from self.select_xsd_nodes(context, self.value)
        return
    else:
        name = self.value
        default_namespace = self.parser.default_namespace
    if self.xsd_types is self.parser.schema:
        for item in context.iter_children_or_self():
            if item.match_name(name, default_namespace):
                yield item
    elif self.xsd_types is None or isinstance(self.xsd_types, AbstractSchemaProxy):
        for item in context.iter_children_or_self():
            if item.match_name(name, default_namespace):
                if item.xsd_type is not None:
                    yield item
                else:
                    xsd_node = self.parser.schema.find(item.path, self.parser.namespaces)
                    if xsd_node is None:
                        self.xsd_types = self.parser.schema
                    elif isinstance(item, AttributeNode):
                        self.xsd_types = {item.name: xsd_node.type}
                    else:
                        self.xsd_types = {item.elem.tag: xsd_node.type}
                    context.item = self.get_typed_node(item)
                    yield context.item
    else:
        for item in context.iter_children_or_self():
            if item.match_name(name, default_namespace):
                if item.xsd_type is not None:
                    yield item
                else:
                    context.item = self.get_typed_node(item)
                    yield context.item