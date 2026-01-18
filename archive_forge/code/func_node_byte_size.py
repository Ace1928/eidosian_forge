import ipaddress
import struct
from ipaddress import IPv4Address, IPv6Address
from os import PathLike
from typing import Any, AnyStr, Dict, IO, List, Optional, Tuple, Union
from maxminddb.const import MODE_AUTO, MODE_MMAP, MODE_FILE, MODE_MEMORY, MODE_FD
from maxminddb.decoder import Decoder
from maxminddb.errors import InvalidDatabaseError
from maxminddb.file import FileBuffer
from maxminddb.types import Record
@property
def node_byte_size(self) -> int:
    """The size of a node in bytes

        :type: int
        """
    return self.record_size // 4