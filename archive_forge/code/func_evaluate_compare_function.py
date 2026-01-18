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
@method(function('compare', nargs=(2, 3), sequence_types=('xs:string?', 'xs:string?', 'xs:string', 'xs:integer?')))
def evaluate_compare_function(self, context=None):
    if self.context is not None:
        context = self.context
    comp1 = self.get_argument(context, 0, cls=str, promote=(AnyURI, UntypedAtomic))
    comp2 = self.get_argument(context, 1, cls=str, promote=(AnyURI, UntypedAtomic))
    if comp1 is None or comp2 is None:
        return []
    if len(self) < 3:
        collation = self.parser.default_collation
    else:
        collation = self.get_argument(context, 2, required=True)
    with CollationManager(collation, self) as manager:
        value = manager.strcoll(comp1, comp2)
    return 0 if not value else 1 if value > 0 else -1