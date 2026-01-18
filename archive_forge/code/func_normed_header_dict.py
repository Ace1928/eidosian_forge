import base64
import hashlib
import os
from typing import Dict, List, Optional, Union
from h11._headers import Headers as H11Headers
from .events import Event
from .typing import Headers
def normed_header_dict(h11_headers: Union[Headers, H11Headers]) -> Dict[bytes, bytes]:
    name_to_values: Dict[bytes, List[bytes]] = {}
    for name, value in h11_headers:
        name_to_values.setdefault(name, []).append(value)
    name_to_normed_value = {}
    for name, values in name_to_values.items():
        name_to_normed_value[name] = b', '.join(values)
    return name_to_normed_value