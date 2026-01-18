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
@method(function('unparsed-text-available', nargs=(1, 2), sequence_types=('xs:string?', 'xs:string', 'xs:boolean')))
def evaluate_unparsed_text_available_function(self, context=None):
    if self.context is not None:
        context = self.context
    href = self.get_argument(context, cls=str)
    if href is None:
        return False
    elif urlsplit(href).fragment:
        return False
    if len(self) > 1:
        encoding = self.get_argument(context, index=1, required=True, cls=str)
    else:
        encoding = 'UTF-8'
    try:
        uri = self.get_absolute_uri(href)
    except ValueError:
        return False
    try:
        codecs.lookup(encoding)
    except LookupError:
        return False
    try:
        with urlopen(uri) as rp:
            stream_reader = codecs.getreader(encoding)(rp)
            for line in stream_reader:
                if any((not is_xml_codepoint(ord(s)) for s in line)):
                    return False
    except URLError:
        return False
    except ValueError:
        if len(self) > 1:
            return False
    else:
        return True
    try:
        with urlopen(uri) as rp:
            stream_reader = codecs.getreader('UTF-16')(rp)
            for line in stream_reader:
                if any((not is_xml_codepoint(ord(s)) for s in line)):
                    return False
    except (ValueError, URLError):
        return False
    else:
        return True