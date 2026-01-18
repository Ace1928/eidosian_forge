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
def load_ssl_context_verify(self) -> ssl.SSLContext:
    """
        Return an SSL context for verified connections.
        """
    if self.trust_env and self.verify is True:
        ca_bundle = get_ca_bundle_from_env()
        if ca_bundle is not None:
            self.verify = ca_bundle
    if isinstance(self.verify, ssl.SSLContext):
        context = self.verify
        self._load_client_certs(context)
        return context
    elif isinstance(self.verify, bool):
        ca_bundle_path = self.DEFAULT_CA_BUNDLE_PATH
    elif Path(self.verify).exists():
        ca_bundle_path = Path(self.verify)
    else:
        raise IOError('Could not find a suitable TLS CA certificate bundle, invalid path: {}'.format(self.verify))
    context = self._create_default_ssl_context()
    context.verify_mode = ssl.CERT_REQUIRED
    context.check_hostname = True
    try:
        context.post_handshake_auth = True
    except AttributeError:
        pass
    try:
        context.hostname_checks_common_name = False
    except AttributeError:
        pass
    if ca_bundle_path.is_file():
        cafile = str(ca_bundle_path)
        logger.debug('load_verify_locations cafile=%r', cafile)
        context.load_verify_locations(cafile=cafile)
    elif ca_bundle_path.is_dir():
        capath = str(ca_bundle_path)
        logger.debug('load_verify_locations capath=%r', capath)
        context.load_verify_locations(capath=capath)
    self._load_client_certs(context)
    return context