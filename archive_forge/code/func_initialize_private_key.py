from __future__ import annotations
import abc
import base64
import json
import os
import tempfile
import typing as t
from ..encoding import (
from ..io import (
from ..config import (
from ..util import (
def initialize_private_key(self) -> str:
    """
        Initialize and publish a new key pair (if needed) and return the private key.
        The private key is cached across ansible-test invocations, so it is only generated and published once per CI job.
        """
    path = os.path.expanduser('~/.ansible-core-ci-private.key')
    if os.path.exists(to_bytes(path)):
        private_key_pem = read_text_file(path)
    else:
        private_key_pem = self.generate_private_key()
        write_text_file(path, private_key_pem)
    return private_key_pem