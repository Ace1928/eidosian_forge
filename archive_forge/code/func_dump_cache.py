from pathlib import Path
import json
from collections.abc import MutableMapping, Mapping
from contextlib import contextmanager
from ase.io.jsonio import read_json, write_json, encode
from ase.utils import opencew
@classmethod
def dump_cache(cls, path, dct):
    cache = cls(path, dct)
    cache._dump_json()
    return cache