import logging
import os
from typing import Any, Awaitable, Dict, List, Optional, Set, Tuple, Union
import zmq
from zmq.error import _check_version
from zmq.utils import z85
from .certs import load_certificates
def configure_gssapi(self, domain: str='*', location: Optional[str]=None) -> None:
    """Configure GSSAPI authentication

        Currently this is a no-op because there is nothing to configure with GSSAPI.
        """