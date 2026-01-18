import datetime
import hashlib
import warnings
from typing import (
from redis.compat import Literal
from redis.exceptions import ConnectionError, DataError, NoScriptError, RedisError
from redis.typing import (
from .helpers import list_or_args
def client_list(self, _type: Union[str, None]=None, client_id: List[EncodableT]=[], **kwargs) -> ResponseT:
    """
        Returns a list of currently connected clients.
        If type of client specified, only that type will be returned.

        :param _type: optional. one of the client types (normal, master,
         replica, pubsub)
        :param client_id: optional. a list of client ids

        For more information see https://redis.io/commands/client-list
        """
    args = []
    if _type is not None:
        client_types = ('normal', 'master', 'replica', 'pubsub')
        if str(_type).lower() not in client_types:
            raise DataError(f'CLIENT LIST _type must be one of {client_types!r}')
        args.append(b'TYPE')
        args.append(_type)
    if not isinstance(client_id, list):
        raise DataError('client_id must be a list')
    if client_id:
        args.append(b'ID')
        args.append(' '.join(client_id))
    return self.execute_command('CLIENT LIST', *args, **kwargs)