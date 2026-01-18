import copy
import random
import socket
import sys
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, Tuple, Union
from aiokeydb.v1.core import CaseInsensitiveDict, PubSub, KeyDB, parse_scan
from aiokeydb.v1.commands import READ_COMMANDS, CommandsParser, KeyDBClusterCommands
from aiokeydb.v1.connection import ConnectionPool, DefaultParser, Encoder, parse_url
from aiokeydb.v1.crc import REDIS_CLUSTER_HASH_SLOTS, key_slot
from aiokeydb.v1.exceptions import (
from aiokeydb.v1.lock import Lock
from aiokeydb.v1.utils import (
def get_keydb_connection(self):
    """
        Get the KeyDB connection of the pubsub connected node.
        """
    if self.node is not None:
        return self.node.keydb_connection