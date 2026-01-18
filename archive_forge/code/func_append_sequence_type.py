import decimal
import os
import re
import codecs
import math
from copy import copy
from itertools import zip_longest
from typing import cast, Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit
from urllib.request import urlopen
from urllib.error import URLError
from ..exceptions import ElementPathError
from ..tdop import MultiLabel
from ..helpers import OCCURRENCE_INDICATORS, EQNAME_PATTERN, \
from ..namespaces import get_expanded_name, split_expanded_name, \
from ..datatypes import xsd10_atomic_types, NumericProxy, QName, Date10, \
from ..sequence_types import is_sequence_type, match_sequence_type
from ..etree import defuse_xml, etree_iter_paths
from ..xpath_nodes import XPathNode, ElementNode, TextNode, AttributeNode, \
from ..tree_builders import get_node_tree
from ..xpath_tokens import XPathFunctionArgType, XPathToken, ValueToken, XPathFunction
from ..serialization import get_serialization_params, serialize_to_xml, serialize_to_json
from ..xpath_context import XPathContext, XPathSchemaContext
from ..regex import translate_pattern, RegexError
from ._xpath30_operators import XPath30Parser
from .xpath30_helpers import UNICODE_DIGIT_PATTERN, DECIMAL_DIGIT_PATTERN, \
def append_sequence_type(tk):
    if tk.symbol == '(' and len(tk) == 1:
        tk = tk[0]
    sequence_type = tk.source
    next_symbol = self.parser.next_token.symbol
    if sequence_type != 'empty-sequence()' and next_symbol in OCCURRENCE_INDICATORS:
        self.parser.advance()
        sequence_type += next_symbol
        tk.occurrence = next_symbol
    if not is_sequence_type(sequence_type, self.parser):
        if 'xs:NMTOKENS' in sequence_type or 'xs:ENTITIES' in sequence_type or 'xs:IDREFS' in sequence_type:
            msg = 'a list type cannot be used in a function signature'
            raise self.error('XPST0051', msg)
        raise self.error('XPST0003', 'a sequence type expected')
    self.sequence_types.append(sequence_type)