import html
import json
import re
import urllib.parse
from tornado.util import unicode_type
import typing
from typing import Union, Any, Optional, Dict, List, Callable
def recursive_unicode(obj: Any) -> Any:
    """Walks a simple data structure, converting byte strings to unicode.

    Supports lists, tuples, and dictionaries.
    """
    if isinstance(obj, dict):
        return dict(((recursive_unicode(k), recursive_unicode(v)) for k, v in obj.items()))
    elif isinstance(obj, list):
        return list((recursive_unicode(i) for i in obj))
    elif isinstance(obj, tuple):
        return tuple((recursive_unicode(i) for i in obj))
    elif isinstance(obj, bytes):
        return to_unicode(obj)
    else:
        return obj