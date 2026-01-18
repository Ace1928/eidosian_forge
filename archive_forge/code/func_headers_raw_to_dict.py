from base64 import b64encode
from typing import Any, List, MutableMapping, Optional, AnyStr, Sequence, Union, Mapping
from w3lib.util import to_bytes, to_unicode
def headers_raw_to_dict(headers_raw: Optional[bytes]) -> Optional[HeadersDictOutput]:
    """
    Convert raw headers (single multi-line bytestring)
    to a dictionary.

    For example:

    >>> import w3lib.http
    >>> w3lib.http.headers_raw_to_dict(b"Content-type: text/html\\n\\rAccept: gzip\\n\\n")   # doctest: +SKIP
    {'Content-type': ['text/html'], 'Accept': ['gzip']}

    Incorrect input:

    >>> w3lib.http.headers_raw_to_dict(b"Content-typt gzip\\n\\n")
    {}
    >>>

    Argument is ``None`` (return ``None``):

    >>> w3lib.http.headers_raw_to_dict(None)
    >>>

    """
    if headers_raw is None:
        return None
    headers = headers_raw.splitlines()
    headers_tuples = [header.split(b':', 1) for header in headers]
    result_dict: HeadersDictOutput = {}
    for header_item in headers_tuples:
        if not len(header_item) == 2:
            continue
        item_key = header_item[0].strip()
        item_value = header_item[1].strip()
        if item_key in result_dict:
            result_dict[item_key].append(item_value)
        else:
            result_dict[item_key] = [item_value]
    return result_dict