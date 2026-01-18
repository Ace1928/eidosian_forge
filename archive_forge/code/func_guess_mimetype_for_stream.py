import base64
import imghdr
from collections import OrderedDict
from os import path
from typing import IO, BinaryIO, NamedTuple, Optional, Tuple
import imagesize
def guess_mimetype_for_stream(stream: IO, default: Optional[str]=None) -> Optional[str]:
    imgtype = imghdr.what(stream)
    if imgtype:
        return 'image/' + imgtype
    else:
        return default