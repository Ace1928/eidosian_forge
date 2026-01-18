import html
import json
import re
import urllib.parse
from tornado.util import unicode_type
import typing
from typing import Union, Any, Optional, Dict, List, Callable
def json_decode(value: Union[str, bytes]) -> Any:
    """Returns Python objects for the given JSON string.

    Supports both `str` and `bytes` inputs. Equvalent to `json.loads`.
    """
    return json.loads(value)