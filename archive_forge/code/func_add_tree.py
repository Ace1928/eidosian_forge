import os
import stat
import struct
import sys
from dataclasses import dataclass
from enum import Enum
from typing import (
from .file import GitFile
from .object_store import iter_tree_contents
from .objects import (
from .pack import ObjectContainer, SHA1Reader, SHA1Writer
def add_tree(path):
    if path in trees:
        return trees[path]
    dirname, basename = pathsplit(path)
    t = add_tree(dirname)
    assert isinstance(basename, bytes)
    newtree = {}
    t[basename] = newtree
    trees[path] = newtree
    return newtree