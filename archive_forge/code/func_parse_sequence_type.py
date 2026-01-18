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
def parse_sequence_type(self) -> 'XPathToken':
    if self.parser.next_token.label in ('kind test', 'sequence type', 'function test'):
        token = self.parser.expression(rbp=85)
    elif self.parser.next_token.symbol == 'Q{':
        token = self.parser.advance().nud()
    elif self.parser.next_token.symbol != '(name)':
        raise self.wrong_syntax()
    else:
        self.parser.advance()
        if self.parser.next_token.symbol == ':':
            left = self.parser.token
            self.parser.advance()
            token = self.parser.token.led(left)
        else:
            token = self.parser.token
        if self.parser.next_token.symbol in ('::', '('):
            raise self.parser.next_token.wrong_syntax()
    next_symbol = self.parser.next_token.symbol
    if token.symbol != 'empty-sequence' and next_symbol in ('?', '*', '+'):
        token.occurrence = next_symbol
        self.parser.advance()
    return token