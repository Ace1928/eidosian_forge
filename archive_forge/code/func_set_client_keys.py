import asyncio
import logging
import sys
from asyncio import AbstractEventLoop
from typing import Callable, List, Optional, Tuple
from datetime import datetime
import zmq.asyncio
from zmq.auth.asyncio import AsyncioAuthenticator
from rpcq._base import to_msgpack, from_msgpack
from rpcq._spec import RPCSpec
from rpcq.messages import RPCRequest
def set_client_keys(self, client_keys: List[bytes]):
    """
        Reset authorized public key list to this set. Avoids the disk read required by configure_curve,
            and allows keys to be managed externally.

        In some cases, keys may be preloaded before the authenticator is started. In this case, we 
            cache those preloaded keys
        """
    if self._authenticator:
        _log.debug(f'Authorizer: Setting client keys to {client_keys}')
        self._authenticator.certs['*'] = {key: True for key in client_keys}
    else:
        _log.debug(f'Authorizer: Preloading client keys to {client_keys}')
        self._preloaded_keys = client_keys