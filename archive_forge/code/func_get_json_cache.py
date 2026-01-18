from pathlib import Path
import json
from collections.abc import MutableMapping, Mapping
from contextlib import contextmanager
from ase.io.jsonio import read_json, write_json, encode
from ase.utils import opencew
def get_json_cache(directory):
    try:
        return CombinedJSONCache.load(directory)
    except FileNotFoundError:
        return MultiFileJSONCache(directory)