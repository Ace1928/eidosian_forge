import dataclasses
import json
import warnings
from dataclasses import dataclass, field
from time import time
from typing import List
from ..utils import logging
def list_field(default=None, metadata=None):
    return field(default_factory=lambda: default, metadata=metadata)