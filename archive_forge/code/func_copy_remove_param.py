from __future__ import annotations
import typing
from urllib.parse import parse_qs, unquote
import idna
from ._types import QueryParamTypes, RawURL, URLTypes
from ._urlparse import urlencode, urlparse
from ._utils import primitive_value_to_str
def copy_remove_param(self, key: str) -> URL:
    return self.copy_with(params=self.params.remove(key))