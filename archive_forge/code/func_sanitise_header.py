from email.header import Header, decode_header, make_header
from email.message import Message
from typing import Any, Dict, List, Union
def sanitise_header(h: Union[Header, str]) -> str:
    if isinstance(h, Header):
        chunks = []
        for bytes, encoding in decode_header(h):
            if encoding == 'unknown-8bit':
                try:
                    bytes.decode('utf-8')
                    encoding = 'utf-8'
                except UnicodeDecodeError:
                    encoding = 'latin1'
            chunks.append((bytes, encoding))
        return str(make_header(chunks))
    return str(h)