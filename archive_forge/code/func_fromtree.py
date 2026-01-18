import collections.abc
import re
from typing import (
import warnings
from io import BytesIO
from datetime import datetime
from base64 import b64encode, b64decode
from numbers import Integral
from types import SimpleNamespace
from functools import singledispatch
from fontTools.misc import etree
from fontTools.misc.textTools import tostr
def fromtree(tree: etree.Element, use_builtin_types: Optional[bool]=None, dict_type: Type[MutableMapping[str, Any]]=dict) -> Any:
    """Convert an XML tree to a plist structure.

    Args:
        tree: An ``etree`` ``Element``.
        use_builtin_types: If True, binary data is deserialized to
            bytes strings. If False, it is wrapped in :py:class:`Data`
            objects. Defaults to True if not provided. Deprecated.
        dict_type: What type to use for dictionaries.

    Returns: An object (usually a dictionary).
    """
    target = PlistTarget(use_builtin_types=use_builtin_types, dict_type=dict_type)
    for action, element in etree.iterwalk(tree, events=('start', 'end')):
        if action == 'start':
            target.start(element.tag, element.attrib)
        elif action == 'end':
            if not len(element):
                target.data(element.text or '')
            target.end(element.tag)
    return target.close()