import json
from typing import IO, Any, Tuple, List
from .parser import Parser
from .symbols import (
def read_object_key(self, key):
    self._key = key