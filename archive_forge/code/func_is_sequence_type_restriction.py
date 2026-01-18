import re
from itertools import zip_longest
from typing import TYPE_CHECKING, cast, Any, Optional
from .exceptions import ElementPathKeyError, xpath_error
from .helpers import OCCURRENCE_INDICATORS, EQNAME_PATTERN, WHITESPACES_PATTERN
from .namespaces import XSD_NAMESPACE, XSD_ERROR, XSD_ANY_SIMPLE_TYPE, XSD_NUMERIC, \
from .datatypes import xsd10_atomic_types, xsd11_atomic_types, AnyAtomicType, \
from .xpath_nodes import XPathNode, DocumentNode, ElementNode, AttributeNode
from . import xpath_tokens
def is_sequence_type_restriction(st1: str, st2: str) -> bool:
    """Returns `True` if st2 is a restriction of st1."""
    st1, st2 = (normalize_sequence_type(st1), normalize_sequence_type(st2))
    if st2 in ('empty-sequence()', 'none') and (st1 in ('empty-sequence()', 'none') or st1.endswith(('?', '*'))):
        return True
    if st1[-1] not in '?+*':
        if st2[-1] in '+*':
            return False
        elif st2[-1] == '?':
            st2 = st2[:-1]
    elif st1[-1] == '+':
        st1 = st1[:-1]
        if st2[-1] in '?*':
            return False
        elif st2[-1] == '+':
            st2 = st2[:-1]
    elif st1[-1] == '*':
        st1 = st1[:-1]
        if st2[-1] in '?+':
            return False
        elif st2[-1] == '*':
            st2 = st2[:-1]
    else:
        st1 = st1[:-1]
        if st2[-1] in '+*':
            return False
        elif st2[-1] == '?':
            st2 = st2[:-1]
    if st1 == st2:
        return True
    elif st1 == 'item()':
        return True
    elif st2 == 'item()':
        return False
    elif st1 == 'node()':
        return st2.startswith(('element(', 'attribute(', 'comment(', 'text(', 'processing-instruction(', 'document(', 'namespace('))
    elif st2 == 'node()':
        return False
    elif st1 == 'xs:anyAtomicType':
        try:
            return issubclass(xsd11_atomic_types[st2[3:]], AnyAtomicType)
        except KeyError:
            return False
    elif st1.startswith('xs:'):
        if st2 == 'xs:anyAtomicType':
            return True
        try:
            return issubclass(xsd11_atomic_types[st2[3:]], xsd11_atomic_types[st1[3:]])
        except KeyError:
            return False
    elif not st1.startswith('function('):
        return False
    if st1 == 'function(*)':
        return st2.startswith('function(')
    parts1 = st1[9:].partition(') as ')
    parts2 = st2[9:].partition(') as ')
    for st1, st2 in zip_longest(parts1[0].split(', '), parts2[0].split(', ')):
        if st1 is None or st2 is None:
            return False
        if not is_sequence_type_restriction(st2, st1):
            return False
    else:
        if not is_sequence_type_restriction(parts1[2], parts2[2]):
            return False
        return True