import sys
import re
import io
import importlib
from typing import cast, Any, Counter, Iterator, Optional, MutableMapping, \
from .protocols import ElementProtocol, DocumentProtocol
import xml.etree.ElementTree as ElementTree
import xml.etree.ElementTree as PyElementTree  # noqa
import xml.etree  # noqa
def is_etree_document(obj: Any) -> bool:
    return hasattr(obj, 'getroot') and hasattr(obj, 'parse') and hasattr(obj, 'iter')