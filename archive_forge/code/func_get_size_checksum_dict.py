import enum
import os
from typing import Optional
from huggingface_hub.utils import insecure_hashlib
from .. import config
from .logging import get_logger
def get_size_checksum_dict(path: str, record_checksum: bool=True) -> dict:
    """Compute the file size and the sha256 checksum of a file"""
    if record_checksum:
        m = insecure_hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                m.update(chunk)
            checksum = m.hexdigest()
    else:
        checksum = None
    return {'num_bytes': os.path.getsize(path), 'checksum': checksum}