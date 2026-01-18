import json
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Type, TypeVar, Union
from .parse import Protocol, load_file, load_str_bytes
from .types import StrBytes
from .typing import display_as_type
def parse_raw_as(type_: Type[T], b: StrBytes, *, content_type: str=None, encoding: str='utf8', proto: Protocol=None, allow_pickle: bool=False, json_loads: Callable[[str], Any]=json.loads, type_name: Optional[NameFactory]=None) -> T:
    obj = load_str_bytes(b, proto=proto, content_type=content_type, encoding=encoding, allow_pickle=allow_pickle, json_loads=json_loads)
    return parse_obj_as(type_, obj, type_name=type_name)