import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def client_tracking(self, on: bool=True, clientid: Union[int, None]=None, prefix: Sequence[KeyT]=[], bcast: bool=False, optin: bool=False, optout: bool=False, noloop: bool=False, **kwargs) -> ResponseT:
    """
        Enables the tracking feature of the Redis server, that is used
        for server assisted client side caching.

        ``on`` indicate for tracking on or tracking off. The dafualt is on.

        ``clientid`` send invalidation messages to the connection with
        the specified ID.

        ``bcast`` enable tracking in broadcasting mode. In this mode
        invalidation messages are reported for all the prefixes
        specified, regardless of the keys requested by the connection.

        ``optin``  when broadcasting is NOT active, normally don't track
        keys in read only commands, unless they are called immediately
        after a CLIENT CACHING yes command.

        ``optout`` when broadcasting is NOT active, normally track keys in
        read only commands, unless they are called immediately after a
        CLIENT CACHING no command.

        ``noloop`` don't send notifications about keys modified by this
        connection itself.

        ``prefix``  for broadcasting, register a given key prefix, so that
        notifications will be provided only for keys starting with this string.

        See https://redis.io/commands/client-tracking
        """
    if len(prefix) != 0 and bcast is False:
        raise DataError('Prefix can only be used with bcast')
    pieces = ['ON'] if on else ['OFF']
    if clientid is not None:
        pieces.extend(['REDIRECT', clientid])
    for p in prefix:
        pieces.extend(['PREFIX', p])
    if bcast:
        pieces.append('BCAST')
    if optin:
        pieces.append('OPTIN')
    if optout:
        pieces.append('OPTOUT')
    if noloop:
        pieces.append('NOLOOP')
    return self.execute_command('CLIENT TRACKING', *pieces)