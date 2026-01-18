import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def client_kill_filter(self, _id: Union[str, None]=None, _type: Union[str, None]=None, addr: Union[str, None]=None, skipme: Union[bool, None]=None, laddr: Union[bool, None]=None, user: str=None, **kwargs) -> ResponseT:
    """
        Disconnects client(s) using a variety of filter options
        :param _id: Kills a client by its unique ID field
        :param _type: Kills a client by type where type is one of 'normal',
        'master', 'slave' or 'pubsub'
        :param addr: Kills a client by its 'address:port'
        :param skipme: If True, then the client calling the command
        will not get killed even if it is identified by one of the filter
        options. If skipme is not provided, the server defaults to skipme=True
        :param laddr: Kills a client by its 'local (bind) address:port'
        :param user: Kills a client for a specific user name
        """
    args = []
    if _type is not None:
        client_types = ('normal', 'master', 'slave', 'pubsub')
        if str(_type).lower() not in client_types:
            raise DataError(f'CLIENT KILL type must be one of {client_types!r}')
        args.extend((b'TYPE', _type))
    if skipme is not None:
        if not isinstance(skipme, bool):
            raise DataError('CLIENT KILL skipme must be a bool')
        if skipme:
            args.extend((b'SKIPME', b'YES'))
        else:
            args.extend((b'SKIPME', b'NO'))
    if _id is not None:
        args.extend((b'ID', _id))
    if addr is not None:
        args.extend((b'ADDR', addr))
    if laddr is not None:
        args.extend((b'LADDR', laddr))
    if user is not None:
        args.extend((b'USER', user))
    if not args:
        raise DataError('CLIENT KILL <filter> <value> ... ... <filter> <value> must specify at least one filter')
    return self.execute_command('CLIENT KILL', *args, **kwargs)