from pathlib import Path
import json
from collections.abc import MutableMapping, Mapping
from contextlib import contextmanager
from ase.io.jsonio import read_json, write_json, encode
from ase.utils import opencew
def strip_empties(self):
    empties = [key for key, value in self.items() if value is None]
    for key in empties:
        del self[key]
    return len(empties)