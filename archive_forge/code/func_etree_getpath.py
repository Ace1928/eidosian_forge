import re
from collections import Counter
from decimal import Decimal
from typing import Any, Callable, Iterator, List, MutableMapping, \
from xml.etree.ElementTree import ParseError
from .exceptions import XMLSchemaValueError, XMLSchemaTypeError
from .names import XSI_SCHEMA_LOCATION, XSI_NONS_SCHEMA_LOCATION
from .aliases import ElementType, NamespacesType, AtomicValueType, NumericValueType
def etree_getpath(elem: ElementType, root: ElementType, namespaces: Optional[NamespacesType]=None, relative: bool=True, add_position: bool=False, parent_path: bool=False) -> Optional[str]:
    """
    Returns the XPath path from *root* to descendant *elem* element.

    :param elem: the descendant element.
    :param root: the root element.
    :param namespaces: an optional mapping from namespace prefix to URI.
    :param relative: returns a relative path.
    :param add_position: add context position to child elements that appear multiple times.
    :param parent_path: if set to `True` returns the parent path. Default is `False`.
    :return: An XPath expression or `None` if *elem* is not a descendant of *root*.
    """
    if relative:
        path = '.'
    elif namespaces:
        path = f'/{get_prefixed_qname(root.tag, namespaces)}'
    else:
        path = f'/{root.tag}'
    if not parent_path:
        for e, path in etree_iterpath(root, elem.tag, path, namespaces, add_position):
            if e is elem:
                return path
    else:
        for e, path in etree_iterpath(root, None, path, namespaces, add_position):
            if elem in e:
                return path
    return None