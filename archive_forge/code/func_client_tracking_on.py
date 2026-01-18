import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def client_tracking_on(self, clientid: Union[int, None]=None, prefix: Sequence[KeyT]=[], bcast: bool=False, optin: bool=False, optout: bool=False, noloop: bool=False) -> ResponseT:
    """
        Turn on the tracking mode.
        For more information about the options look at client_tracking func.

        See https://redis.io/commands/client-tracking
        """
    return self.client_tracking(True, clientid, prefix, bcast, optin, optout, noloop)