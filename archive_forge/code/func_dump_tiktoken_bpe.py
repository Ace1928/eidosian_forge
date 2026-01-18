from __future__ import annotations
import base64
import hashlib
import json
import os
import tempfile
import uuid
from typing import Optional
import requests
def dump_tiktoken_bpe(bpe_ranks: dict[bytes, int], tiktoken_bpe_file: str) -> None:
    try:
        import blobfile
    except ImportError as e:
        raise ImportError('blobfile is not installed. Please install it by running `pip install blobfile`.') from e
    with blobfile.BlobFile(tiktoken_bpe_file, 'wb') as f:
        for token, rank in sorted(bpe_ranks.items(), key=lambda x: x[1]):
            f.write(base64.b64encode(token) + b' ' + str(rank).encode() + b'\n')