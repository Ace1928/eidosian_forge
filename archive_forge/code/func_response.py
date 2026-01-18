from __future__ import annotations
import dataclasses
from typing import Any, BinaryIO, Dict, Optional, TYPE_CHECKING, Union
import requests
from requests import PreparedRequest
from requests.auth import AuthBase
from requests.structures import CaseInsensitiveDict
from requests_toolbelt.multipart.encoder import MultipartEncoder  # type: ignore
from . import protocol
@property
def response(self) -> requests.Response:
    return self._response