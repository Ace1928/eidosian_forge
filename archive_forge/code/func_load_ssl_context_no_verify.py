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
def load_ssl_context_no_verify(self) -> ssl.SSLContext:
    """
        Return an SSL context for unverified connections.
        """
    context = self._create_default_ssl_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    self._load_client_certs(context)
    return context