import logging
from io import BytesIO
from typing import BinaryIO, Iterator, List, Optional, cast
def lzwdecode(data: bytes) -> bytes:
    fp = BytesIO(data)
    s = LZWDecoder(fp).run()
    return b''.join(s)