from __future__ import annotations
import logging
import os
import ssl
import typing
from pathlib import Path
import certifi
from ._compat import set_minimum_tls_version_1_2
from ._models import Headers
from ._types import CertTypes, HeaderTypes, TimeoutTypes, URLTypes, VerifyTypes
from ._urls import URL
from ._utils import get_ca_bundle_from_env
@property
def raw_auth(self) -> tuple[bytes, bytes] | None:
    return None if self.auth is None else (self.auth[0].encode('utf-8'), self.auth[1].encode('utf-8'))