import re
from itertools import zip_longest
from typing import TYPE_CHECKING, cast, Any, Optional
from .exceptions import ElementPathKeyError, xpath_error
from .helpers import OCCURRENCE_INDICATORS, EQNAME_PATTERN, WHITESPACES_PATTERN
from .namespaces import XSD_NAMESPACE, XSD_ERROR, XSD_ANY_SIMPLE_TYPE, XSD_NUMERIC, \
from .datatypes import xsd10_atomic_types, xsd11_atomic_types, AnyAtomicType, \
from .xpath_nodes import XPathNode, DocumentNode, ElementNode, AttributeNode
from . import xpath_tokens
def normalize_sequence_type(sequence_type: str) -> str:
    sequence_type = WHITESPACES_PATTERN.sub(' ', sequence_type).strip()
    sequence_type = SEQUENCE_TYPE_PATTERN.sub('\\1', sequence_type)
    return sequence_type.replace(',', ', ').replace(')as', ') as')