import sys
import pickle
import struct
import pprint
import zipfile
import fnmatch
from typing import Any, IO, BinaryIO, Union
def fake_new(self, *args):
    return FakeObject(self.module, self.name, args[1:])