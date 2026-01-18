import base64
import imghdr
from collections import OrderedDict
from os import path
from typing import IO, BinaryIO, NamedTuple, Optional, Tuple
import imagesize
def parse_data_uri(uri: str) -> Optional[DataURI]:
    if not uri.startswith('data:'):
        return None
    mimetype = 'text/plain'
    charset = 'US-ASCII'
    properties, data = uri[5:].split(',', 1)
    for prop in properties.split(';'):
        if prop == 'base64':
            pass
        elif prop.startswith('charset='):
            charset = prop[8:]
        elif prop:
            mimetype = prop
    image_data = base64.b64decode(data)
    return DataURI(mimetype, charset, image_data)