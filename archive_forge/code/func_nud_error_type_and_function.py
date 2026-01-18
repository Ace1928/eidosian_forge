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
@method('error')
def nud_error_type_and_function(self):
    self.clear()
    try:
        self.parser.advance('(')
        if self.namespace == XSD_NAMESPACE:
            self.label = 'constructor function'
            self.nargs = 1
            if self.parser.xsd_version == '1.0':
                raise self.error('XPST0051', 'xs:error is not defined with XSD 1.0')
            self.append(self.parser.expression(5))
        else:
            self.label = 'function'
            for k in range(3):
                if self.parser.next_token.symbol == ')':
                    break
                self.append(self.parser.expression(5))
                if self.parser.next_token.symbol == ')':
                    break
                self.parser.advance(',')
        self.parser.advance(')')
    except SyntaxError:
        raise self.error('XPST0017') from None
    self.value = None
    return self