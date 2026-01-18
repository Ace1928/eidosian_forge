import base64
import logging
import os
import socket
import typing
from spnego._context import (
from spnego._credential import (
from spnego._ntlm_raw.crypto import (
from spnego._ntlm_raw.messages import (
from spnego._ntlm_raw.security import seal, sign
from spnego._text import to_text
from spnego.channel_bindings import GssChannelBindings
from spnego.exceptions import (
from spnego.iov import BufferType, IOVResBuffer
def store_lines(text: str) -> typing.Iterator[typing.Tuple[str, str, typing.Optional[str], typing.Optional[bytes], typing.Optional[bytes]]]:
    for line in text.splitlines():
        line_split = line.split(':')
        if len(line_split) == 3:
            yield (line_split[0], line_split[1], line_split[2], None, None)
        elif len(line_split) == 6:
            domain_entry, user_entry = split_username(line_split[0])
            lm_entry = base64.b16decode(line_split[2].upper())
            nt_entry = base64.b16decode(line_split[3].upper())
            yield (domain_entry or '', user_entry or '', None, lm_entry, nt_entry)