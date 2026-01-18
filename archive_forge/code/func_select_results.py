import decimal
import math
from copy import copy
from decimal import Decimal
from itertools import product
from typing import TYPE_CHECKING, cast, Dict, Optional, List, Tuple, \
import urllib.parse
from .exceptions import ElementPathError, ElementPathValueError, \
from .helpers import ordinal, get_double, split_function_test
from .etree import is_etree_element, is_etree_document
from .namespaces import XSD_NAMESPACE, XPATH_FUNCTIONS_NAMESPACE, \
from .tree_builders import get_node_tree
from .xpath_nodes import XPathNode, ElementNode, AttributeNode, \
from .datatypes import xsd10_atomic_types, AbstractDateTime, AnyURI, \
from .protocols import ElementProtocol, DocumentProtocol, XsdAttributeProtocol, \
from .sequence_types import is_sequence_type_restriction, match_sequence_type
from .schema_proxy import AbstractSchemaProxy
from .tdop import Token, MultiLabel
from .xpath_context import XPathContext, XPathSchemaContext
def select_results(self, context: ContextArgType) -> Iterator[XPathResultType]:
    """
        Generates formatted XPath results.

        :param context: the XPath dynamic context.
        """
    if context is None:
        yield from self.select(context)
    else:
        self.parser.check_variables(context.variables)
        for result in self.select(context):
            if not isinstance(result, XPathNode):
                yield result
            elif isinstance(result, NamespaceNode):
                if self.parser.compatibility_mode:
                    yield (result.prefix, result.uri)
                else:
                    yield result.uri
            elif isinstance(result, DocumentNode):
                if result.is_extended():
                    yield result
                elif result is context.root or result is not context.document:
                    yield result.value
            else:
                yield result.value