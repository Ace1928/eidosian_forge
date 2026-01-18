import logging
import os
from typing import Any, Awaitable, Dict, List, Optional, Set, Tuple, Union
import zmq
from zmq.error import _check_version
from zmq.utils import z85
from .certs import load_certificates
def configure_plain(self, domain: str='*', passwords: Optional[Dict[str, str]]=None) -> None:
    """Configure PLAIN authentication for a given domain.

        PLAIN authentication uses a plain-text password file.
        To cover all domains, use "*".
        You can modify the password file at any time; it is reloaded automatically.
        """
    if passwords:
        self.passwords[domain] = passwords
    self.log.debug('Configure plain: %s', domain)