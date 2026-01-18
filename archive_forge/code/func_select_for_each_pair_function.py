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
@method(function('for-each-pair', nargs=3, sequence_types=('item()*', 'item()*', 'function(item(), item()) as item()*', 'item()*')))
def select_for_each_pair_function(self, context=None):
    func = self[2][1] if self[2].symbol == ':' else self[2]
    if not isinstance(func, XPathFunction):
        func = self.get_argument(context, index=2, cls=XPathFunction, required=True)
    if not isinstance(func, XPathFunction):
        raise self.error('XPTY0004', 'invalid type for 3rd argument {!r}'.format(func))
    elif func.arity != 2:
        raise self.error('XPTY0004', 'function arity of 3rd argument must be 2')
    for item1, item2 in zip(self[0].select(copy(context)), self[1].select(copy(context))):
        result = func(item1, item2, context=context)
        if isinstance(result, list):
            yield from result
        else:
            yield result