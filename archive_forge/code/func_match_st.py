import re
from itertools import zip_longest
from typing import TYPE_CHECKING, cast, Any, Optional
from .exceptions import ElementPathKeyError, xpath_error
from .helpers import OCCURRENCE_INDICATORS, EQNAME_PATTERN, WHITESPACES_PATTERN
from .namespaces import XSD_NAMESPACE, XSD_ERROR, XSD_ANY_SIMPLE_TYPE, XSD_NUMERIC, \
from .datatypes import xsd10_atomic_types, xsd11_atomic_types, AnyAtomicType, \
from .xpath_nodes import XPathNode, DocumentNode, ElementNode, AttributeNode
from . import xpath_tokens
def match_st(v: Any, st: str, occurrence: Optional[str]=None) -> bool:
    if st[-1] in OCCURRENCE_INDICATORS and ') as ' not in st:
        return match_st(v, st[:-1], st[-1])
    elif v is None or (isinstance(v, list) and v == []):
        return st in ('empty-sequence()', 'none') or occurrence in ('?', '*')
    elif st in ('empty-sequence()', 'none'):
        return False
    elif isinstance(v, list):
        if len(v) == 1:
            return match_st(v[0], st)
        elif occurrence is None or occurrence == '?':
            return False
        else:
            return all((match_st(x, st) for x in v))
    elif st == 'item()':
        return isinstance(v, (XPathNode, AnyAtomicType, list, xpath_tokens.XPathFunction))
    elif st == 'numeric' or st == 'xs:numeric':
        return isinstance(v, NumericProxy)
    elif st.startswith('function('):
        if not isinstance(v, xpath_tokens.XPathFunction):
            return False
        return v.match_function_test(st)
    elif st.startswith('array('):
        if not isinstance(v, xpath_tokens.XPathArray):
            return False
        if st == 'array(*)':
            return True
        item_st = st[6:-1]
        return all((match_st(x, item_st) for x in v.items()))
    elif st.startswith('map('):
        if not isinstance(v, xpath_tokens.XPathMap):
            return False
        if st == 'map(*)':
            return True
        key_st, _, value_st = st[4:-1].partition(', ')
        if key_st.endswith(('+', '*')):
            raise xpath_error('XPST0003', 'no multiples occurs for a map key')
        return all((match_st(k, key_st) and match_st(v, value_st) for k, v in v.items()))
    if isinstance(v, XPathNode):
        value_kind = v.kind
    elif '(' in st:
        return False
    elif not strict and st == 'xs:anyURI' and isinstance(v, str):
        return True
    else:
        try:
            return is_instance(v, st, parser)
        except (KeyError, ValueError):
            raise xpath_error('XPST0051')
    if st == 'node()':
        return True
    elif not st.startswith(value_kind) or not st.endswith(')'):
        return False
    elif st == f'{value_kind}()':
        return True
    elif value_kind == 'document':
        element_test = st[14:-1]
        if not element_test:
            return True
        document = cast(DocumentNode, v)
        return any((match_st(e, element_test) for e in document if isinstance(e, ElementNode)))
    elif value_kind not in ('element', 'attribute'):
        return False
    _, params = st[:-1].split('(')
    if ', ' not in st:
        name = params
    else:
        name, type_name = params.rsplit(', ', 1)
        if type_name.endswith('?'):
            type_name = type_name[:-1]
        elif isinstance(v, ElementNode) and v.nilled:
            return False
        if type_name == 'xs:untyped':
            if isinstance(v, (ElementNode, AttributeNode)) and v.xsd_type is not None:
                return False
        else:
            try:
                if not is_instance(v.typed_value, type_name, parser):
                    return False
            except (KeyError, ValueError):
                raise xpath_error('XPST0051')
    if name == '*':
        return True
    try:
        exp_name = get_expanded_name(name, parser.namespaces)
    except (KeyError, ValueError):
        return False
    except AttributeError:
        return True if v.name == name else False
    else:
        return True if v.name == exp_name else False