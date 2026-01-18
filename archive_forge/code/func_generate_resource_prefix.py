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
@abc.abstractmethod
def generate_resource_prefix(self) -> str:
    """Return a resource prefix specific to this CI provider."""