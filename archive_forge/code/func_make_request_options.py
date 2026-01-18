from __future__ import annotations
import json
import time
import uuid
import email
import asyncio
import inspect
import logging
import platform
import warnings
import email.utils
from types import TracebackType
from random import random
from typing import (
from functools import lru_cache
from typing_extensions import Literal, override, get_origin
import anyio
import httpx
import distro
import pydantic
from httpx import URL, Limits
from pydantic import PrivateAttr
from . import _exceptions
from ._qs import Querystring
from ._files import to_httpx_files, async_to_httpx_files
from ._types import (
from ._utils import is_dict, is_list, is_given, is_mapping
from ._compat import model_copy, model_dump
from ._models import GenericModel, FinalRequestOptions, validate_type, construct_type
from ._response import (
from ._constants import (
from ._streaming import Stream, SSEDecoder, AsyncStream, SSEBytesDecoder
from ._exceptions import (
from ._legacy_response import LegacyAPIResponse
def make_request_options(*, query: Query | None=None, extra_headers: Headers | None=None, extra_query: Query | None=None, extra_body: Body | None=None, idempotency_key: str | None=None, timeout: float | httpx.Timeout | None | NotGiven=NOT_GIVEN, post_parser: PostParser | NotGiven=NOT_GIVEN) -> RequestOptions:
    """Create a dict of type RequestOptions without keys of NotGiven values."""
    options: RequestOptions = {}
    if extra_headers is not None:
        options['headers'] = extra_headers
    if extra_body is not None:
        options['extra_json'] = cast(AnyMapping, extra_body)
    if query is not None:
        options['params'] = query
    if extra_query is not None:
        options['params'] = {**options.get('params', {}), **extra_query}
    if not isinstance(timeout, NotGiven):
        options['timeout'] = timeout
    if idempotency_key is not None:
        options['idempotency_key'] = idempotency_key
    if is_given(post_parser):
        options['post_parser'] = post_parser
    return options