import sys
import re
import io
import importlib
from typing import cast, Any, Counter, Iterator, Optional, MutableMapping, \
from .protocols import ElementProtocol, DocumentProtocol
import xml.etree.ElementTree as ElementTree
import xml.etree.ElementTree as PyElementTree  # noqa
import xml.etree  # noqa
def etree_iter_strings(elem: Union[DocumentProtocol, ElementProtocol], normalize: bool=False) -> Iterator[str]:
    e: ElementProtocol
    if normalize:
        if hasattr(elem, 'getroot'):
            root = cast(DocumentProtocol, elem).getroot()
            if root is None:
                return
        else:
            root = elem
        for e in elem.iter():
            if callable(e.tag):
                continue
            if e.text is not None:
                yield (e.text.strip() if e is root else e.text)
            if e.tail is not None and e is not root:
                yield (e.tail.strip() if e in root else e.tail)
    else:
        for e in elem.iter():
            if callable(e.tag):
                continue
            if e.text is not None:
                yield e.text
            if e.tail is not None and e is not elem:
                yield e.tail