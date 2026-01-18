import re
from collections import Counter
from decimal import Decimal
from typing import Any, Callable, Iterator, List, MutableMapping, \
from xml.etree.ElementTree import ParseError
from .exceptions import XMLSchemaValueError, XMLSchemaTypeError
from .names import XSI_SCHEMA_LOCATION, XSI_NONS_SCHEMA_LOCATION
from .aliases import ElementType, NamespacesType, AtomicValueType, NumericValueType
def prune_etree(root: ElementType, selector: Callable[[ElementType], bool]) -> Optional[bool]:
    """
    Removes from a tree structure the elements that verify the selector
    function. The checking and eventual removals are performed using a
    breadth-first visit method.

    :param root: the root element of the tree.
    :param selector: the single argument function to apply on each visited node.
    :return: `True` if the root node verify the selector function, `None` otherwise.
    """

    def _prune_subtree(elem: ElementType) -> None:
        for child in elem[:]:
            if selector(child):
                elem.remove(child)
        for child in elem:
            _prune_subtree(child)
    if selector(root):
        del root[:]
        return True
    _prune_subtree(root)
    return None