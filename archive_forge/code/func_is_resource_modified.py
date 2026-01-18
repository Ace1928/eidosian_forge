from __future__ import annotations
import re
import typing as t
from datetime import datetime
from .._internal import _dt_as_utc
from ..http import generate_etag
from ..http import parse_date
from ..http import parse_etags
from ..http import parse_if_range_header
from ..http import unquote_etag
from .. import datastructures as ds
def is_resource_modified(http_range: str | None=None, http_if_range: str | None=None, http_if_modified_since: str | None=None, http_if_none_match: str | None=None, http_if_match: str | None=None, etag: str | None=None, data: bytes | None=None, last_modified: datetime | str | None=None, ignore_if_range: bool=True) -> bool:
    """Convenience method for conditional requests.
    :param http_range: Range HTTP header
    :param http_if_range: If-Range HTTP header
    :param http_if_modified_since: If-Modified-Since HTTP header
    :param http_if_none_match: If-None-Match HTTP header
    :param http_if_match: If-Match HTTP header
    :param etag: the etag for the response for comparison.
    :param data: or alternatively the data of the response to automatically
                 generate an etag using :func:`generate_etag`.
    :param last_modified: an optional date of the last modification.
    :param ignore_if_range: If `False`, `If-Range` header will be taken into
                            account.
    :return: `True` if the resource was modified, otherwise `False`.

    .. versionadded:: 2.2
    """
    if etag is None and data is not None:
        etag = generate_etag(data)
    elif data is not None:
        raise TypeError('both data and etag given')
    unmodified = False
    if isinstance(last_modified, str):
        last_modified = parse_date(last_modified)
    if last_modified is not None:
        last_modified = _dt_as_utc(last_modified.replace(microsecond=0))
    if_range = None
    if not ignore_if_range and http_range is not None:
        if_range = parse_if_range_header(http_if_range)
    if if_range is not None and if_range.date is not None:
        modified_since: datetime | None = if_range.date
    else:
        modified_since = parse_date(http_if_modified_since)
    if modified_since and last_modified and (last_modified <= modified_since):
        unmodified = True
    if etag:
        etag, _ = unquote_etag(etag)
        etag = t.cast(str, etag)
        if if_range is not None and if_range.etag is not None:
            unmodified = parse_etags(if_range.etag).contains(etag)
        else:
            if_none_match = parse_etags(http_if_none_match)
            if if_none_match:
                unmodified = if_none_match.contains_weak(etag)
            if_match = parse_etags(http_if_match)
            if if_match:
                unmodified = not if_match.is_strong(etag)
    return not unmodified