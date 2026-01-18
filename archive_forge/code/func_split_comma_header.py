import base64
import hashlib
import os
from typing import Dict, List, Optional, Union
from h11._headers import Headers as H11Headers
from .events import Event
from .typing import Headers
def split_comma_header(value: bytes) -> List[str]:
    return [piece.decode('ascii').strip() for piece in value.split(b',')]