from __future__ import annotations
import re
import json
import copy
import contextlib
import operator
from abc import ABC
from pydantic import BaseModel
from typing import Any, Dict, List, Optional, Union, Callable, Tuple, TYPE_CHECKING
from lazyops.utils import logger
from lazyops.utils.lazy import lazy_import
from lazyops.libs.fastapi_utils.types.user_roles import UserRole
def json_replace(schema: Dict[str, Any], key: str, value: Optional[str]=None, verbose: Optional[bool]=False, key_start: Optional[str]=KEY_START, key_end: Optional[str]=KEY_END, sep_char: Optional[str]=KEY_SEP) -> Dict[str, Any]:
    """
    Replace the json
    """
    plaintext = json.dumps(schema, ensure_ascii=False)
    if f'{key_start} {key}' not in plaintext:
        return schema
    while f'{key_start} {key}' in plaintext:
        src_value = value
        start = plaintext.index(f'{key_start} {key}')
        end = plaintext.index(f'{key_end}', start)
        segment = plaintext[start:end + len(key_end)].strip()
        if not src_value and sep_char in segment:
            src_value = segment.split(sep_char, 1)[-1].rsplit(' ', 1)[0].strip()
        plaintext = plaintext.replace(segment, src_value)
        if verbose:
            logger.info(f'Replaced `{segment}` -> `{src_value}`', colored=True, prefix='|g|OpenAPI|e|')
        if f'{key_start} {key}' not in plaintext:
            break
    schema = json.loads(plaintext)
    return schema