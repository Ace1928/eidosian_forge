from gitdb.db.base import (
from gitdb.db.loose import LooseObjectDB
from gitdb.db.pack import PackedDB
from gitdb.db.ref import ReferenceDB
from gitdb.exc import InvalidDBRoot
import os
def set_ostream(self, ostream):
    return self._loose_db.set_ostream(ostream)