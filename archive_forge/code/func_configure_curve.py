import logging
import os
from typing import Any, Awaitable, Dict, List, Optional, Set, Tuple, Union
import zmq
from zmq.error import _check_version
from zmq.utils import z85
from .certs import load_certificates
def configure_curve(self, domain: str='*', location: Union[str, os.PathLike]='.') -> None:
    """Configure CURVE authentication for a given domain.

        CURVE authentication uses a directory that holds all public client certificates,
        i.e. their public keys.

        To cover all domains, use "*".

        You can add and remove certificates in that directory at any time. configure_curve must be called
        every time certificates are added or removed, in order to update the Authenticator's state

        To allow all client keys without checking, specify CURVE_ALLOW_ANY for the location.
        """
    self.log.debug('Configure curve: %s[%s]', domain, location)
    if location == CURVE_ALLOW_ANY:
        self.allow_any = True
    else:
        self.allow_any = False
        try:
            self.certs[domain] = load_certificates(location)
        except Exception as e:
            self.log.error('Failed to load CURVE certs from %s: %s', location, e)