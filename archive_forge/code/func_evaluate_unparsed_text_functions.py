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
@method(function('unparsed-text', nargs=(1, 2), sequence_types=('xs:string?', 'xs:string', 'xs:string?')))
@method(function('unparsed-text-lines', nargs=(1, 2), sequence_types=('xs:string?', 'xs:string', 'xs:string*')))
def evaluate_unparsed_text_functions(self, context=None):
    if self.context is not None:
        context = self.context
    href = self.get_argument(context, cls=str)
    if href is None:
        return []
    elif urlsplit(href).fragment:
        raise self.error('FOUT1170')
    if len(self) > 1:
        encoding = self.get_argument(context, index=1, required=True, cls=str)
    else:
        encoding = 'UTF-8'
    try:
        uri = self.get_absolute_uri(href)
    except ValueError:
        raise self.error('FOUT1170') from None
    try:
        codecs.lookup(encoding)
    except LookupError:
        raise self.error('FOUT1190') from None
    if context is not None and uri in context.text_resources:
        text = context.text_resources[uri]
    else:
        try:
            with urlopen(uri) as rp:
                stream_reader = codecs.getreader(encoding)(rp)
                text = stream_reader.read()
        except URLError as err:
            raise self.error('FOUT1170', err) from None
        except ValueError as err:
            if len(self) > 1:
                raise self.error('FOUT1190', err) from None
            try:
                with urlopen(uri) as rp:
                    stream_reader = codecs.getreader('UTF-16')(rp)
                    text = stream_reader.read()
            except URLError as err:
                raise self.error('FOUT1170', err) from None
            except ValueError as err:
                raise self.error('FOUT1190', err) from None
        if context is not None:
            context.text_resources[uri] = text
    if not all((is_xml_codepoint(ord(s)) for s in text)):
        raise self.error('FOUT1190')
    text = text.lstrip('\ufeff')
    if self.symbol == 'unparsed-text-lines':
        lines = XML_NEWLINES_PATTERN.split(text)
        return lines[:-1] if lines[-1] == '' else lines
    return text