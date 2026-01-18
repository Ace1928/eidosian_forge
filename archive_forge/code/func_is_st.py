import re
from itertools import zip_longest
from typing import TYPE_CHECKING, cast, Any, Optional
from .exceptions import ElementPathKeyError, xpath_error
from .helpers import OCCURRENCE_INDICATORS, EQNAME_PATTERN, WHITESPACES_PATTERN
from .namespaces import XSD_NAMESPACE, XSD_ERROR, XSD_ANY_SIMPLE_TYPE, XSD_NUMERIC, \
from .datatypes import xsd10_atomic_types, xsd11_atomic_types, AnyAtomicType, \
from .xpath_nodes import XPathNode, DocumentNode, ElementNode, AttributeNode
from . import xpath_tokens
def is_st(st: str) -> bool:
    if not st:
        return False
    elif st == 'empty-sequence()' or st == 'none':
        return True
    elif st[-1] in OCCURRENCE_INDICATORS:
        st = st[:-1]
    if st in COMMON_SEQUENCE_TYPES:
        return True
    elif st.startswith(('map(', 'array(')):
        if parser and parser.version < '3.1' or not st.endswith(')'):
            return False
        if st in ('map(*)', 'array(*)'):
            return True
        if st.startswith('map('):
            key_type, _, value_type = st[4:-1].partition(', ')
            return key_type.startswith('xs:') and (not key_type.endswith(('+', '*'))) and is_st(key_type) and is_st(key_type)
        else:
            return is_st(st[6:-1])
    elif st.startswith('element(') and st.endswith(')'):
        if ',' not in st:
            return EQNAME_PATTERN.match(st[8:-1]) is not None
        try:
            arg1, arg2 = st[8:-1].split(', ')
        except ValueError:
            return False
        else:
            return (arg1 == '*' or EQNAME_PATTERN.match(arg1) is not None) and EQNAME_PATTERN.match(arg2) is not None
    elif st.startswith('document-node(') and st.endswith(')'):
        if not st.startswith('document-node(element('):
            return False
        return is_st(st[14:-1])
    elif st.startswith('function('):
        if parser and parser.version < '3.0':
            return False
        elif st == 'function(*)':
            return True
        elif ' as ' in st:
            pass
        elif not st.endswith(')'):
            return False
        else:
            return is_st(st[9:-1])
        st, return_type = st.rsplit(' as ', 1)
        if not is_st(return_type):
            return False
        elif st == 'function()':
            return True
        st = st[9:-1]
        if st.endswith(', ...'):
            st = st[:-5]
        if 'function(' not in st:
            return all((is_st(x) for x in st.split(', ')))
        elif st.startswith('function(*)') and 'function(' not in st[11:]:
            return all((is_st(x) for x in st.split(', ')))
        k = st.index('function(')
        if not is_st(st[k:]):
            return False
        return all((is_st(x) for x in st[:k].split(', ') if x))
    elif QName.pattern.match(st) is None:
        return False
    if parser is None:
        return False
    try:
        is_instance(None, st, parser)
    except (KeyError, ValueError):
        return False
    else:
        return True