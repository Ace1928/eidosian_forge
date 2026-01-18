from __future__ import annotations
import base64
import hashlib
import json
import os
import tempfile
import uuid
from typing import Optional
import requests
def decode_data_gym(value: str) -> bytes:
    return bytes((data_gym_byte_to_byte[b] for b in value))