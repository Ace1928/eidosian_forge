import re
from typing import Any, Dict, List, Type, Union, Iterator
from xml.etree.ElementTree import Element
from ..helpers import get_namespace, get_qname
def iter_nested_items(items: Union[Dict[Any, Any], List[Any]], dict_class: Type[Dict[Any, Any]]=dict, list_class: Type[List[Any]]=list) -> Iterator[Any]:
    """Iterates a nested object composed by lists and dictionaries."""
    if isinstance(items, dict_class):
        for k, v in items.items():
            yield from iter_nested_items(v, dict_class, list_class)
    elif isinstance(items, list_class):
        for item in items:
            yield from iter_nested_items(item, dict_class, list_class)
    elif isinstance(items, dict):
        raise TypeError('%r: is a dict() instead of %r.' % (items, dict_class))
    elif isinstance(items, list):
        raise TypeError('%r: is a list() instead of %r.' % (items, list_class))
    else:
        yield items