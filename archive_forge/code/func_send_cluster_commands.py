import random
import socket
import sys
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from redis._parsers import CommandsParser, Encoder
from redis._parsers.helpers import parse_scan
from redis.backoff import default_backoff
from redis.client import CaseInsensitiveDict, PubSub, Redis
from redis.commands import READ_COMMANDS, RedisClusterCommands
from redis.commands.helpers import list_or_args
from redis.connection import ConnectionPool, DefaultParser, parse_url
from redis.crc import REDIS_CLUSTER_HASH_SLOTS, key_slot
from redis.exceptions import (
from redis.lock import Lock
from redis.retry import Retry
from redis.utils import (
def send_cluster_commands(self, stack, raise_on_error=True, allow_redirections=True):
    """
        Wrapper for CLUSTERDOWN error handling.

        If the cluster reports it is down it is assumed that:
         - connection_pool was disconnected
         - connection_pool was reseted
         - refereh_table_asap set to True

        It will try the number of times specified by
        the config option "self.cluster_error_retry_attempts"
        which defaults to 3 unless manually configured.

        If it reaches the number of times, the command will
        raises ClusterDownException.
        """
    if not stack:
        return []
    retry_attempts = self.cluster_error_retry_attempts
    while True:
        try:
            return self._send_cluster_commands(stack, raise_on_error=raise_on_error, allow_redirections=allow_redirections)
        except (ClusterDownError, ConnectionError) as e:
            if retry_attempts > 0:
                retry_attempts -= 1
                pass
            else:
                raise e