from itertools import count
from typing import Dict, Iterator, List, TypeVar
from attrs import Factory, define
from twisted.protocols.amp import AMP, Command, Integer, String as Bytes
def chunk(data: bytes, chunkSize: int) -> Iterator[bytes]:
    """
    Break a byte string into pieces of no more than ``chunkSize`` length.

    @param data: The byte string.

    @param chunkSize: The maximum length of the resulting pieces.  All pieces
        except possibly the last will be this length.

    @return: The pieces.
    """
    pos = 0
    while pos < len(data):
        yield data[pos:pos + chunkSize]
        pos += chunkSize