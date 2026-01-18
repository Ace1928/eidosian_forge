import math
import datetime
import time
import re
import os.path
import unicodedata
from copy import copy
from decimal import Decimal, DecimalException
from string import ascii_letters
from urllib.parse import urlsplit, quote as urllib_quote
from ..exceptions import ElementPathValueError
from ..helpers import QNAME_PATTERN, is_idrefs, is_xml_codepoint, round_number
from ..datatypes import DateTime10, DateTime, Date10, Date, Float10, \
from ..namespaces import XML_NAMESPACE, get_namespace, split_expanded_name, \
from ..compare import deep_equal
from ..sequence_types import match_sequence_type
from ..xpath_context import XPathSchemaContext
from ..xpath_nodes import XPathNode, DocumentNode, ElementNode, SchemaElementNode
from ..xpath_tokens import XPathFunction
from ..regex import RegexError, translate_pattern
from ..collations import CollationManager
from ._xpath2_operators import XPath2Parser
@method(function('idref', nargs=(1, 2), sequence_types=('xs:string*', 'node()', 'node()*')))
def select_idref_function(self, context=None):
    if self.context is not None:
        context = self.context
    ids = [x for x in self[0].select(context=copy(context))]
    node = self.get_argument(context, index=1, default_to_context=True)
    if isinstance(context, XPathSchemaContext):
        return
    elif context is None or node is not context.item:
        pass
    elif context.item is None:
        node = context.root
    if not isinstance(node, XPathNode):
        raise self.error('XPTY0004')
    elif isinstance(node, (ElementNode, DocumentNode)):
        for element in filter(lambda x: isinstance(x, ElementNode), node.iter_descendants()):
            text = element.elem.text
            if text and is_idrefs(text) and any((v in text.split() for x in ids for v in x.split())):
                yield element
                continue
            for attr in element.attributes:
                if attr.name != XML_ID and any((v in attr.value.split() for x in ids for v in x.split())):
                    yield element
                    break