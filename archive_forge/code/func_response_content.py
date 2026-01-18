import email.message
import logging
import pathlib
import traceback
import urllib.parse
import warnings
from typing import Any, Callable, Dict, Iterator, Literal, Optional, Tuple, Type, Union
import requests
from gitlab import types
def response_content(response: requests.Response, streamed: bool, action: Optional[Callable[[bytes], None]], chunk_size: int, *, iterator: bool) -> Optional[Union[bytes, Iterator[Any]]]:
    if iterator:
        return response.iter_content(chunk_size=chunk_size)
    if streamed is False:
        return response.content
    if action is None:
        action = _StdoutStream()
    for chunk in response.iter_content(chunk_size=chunk_size):
        if chunk:
            action(chunk)
    return None