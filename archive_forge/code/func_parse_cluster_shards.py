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
def parse_cluster_shards(resp, **options):
    """
    Parse CLUSTER SHARDS response.
    """
    if isinstance(resp[0], dict):
        return resp
    shards = []
    for x in resp:
        shard = {'slots': [], 'nodes': []}
        for i in range(0, len(x[1]), 2):
            shard['slots'].append((x[1][i], x[1][i + 1]))
        nodes = x[3]
        for node in nodes:
            dict_node = {}
            for i in range(0, len(node), 2):
                dict_node[node[i]] = node[i + 1]
            shard['nodes'].append(dict_node)
        shards.append(shard)
    return shards