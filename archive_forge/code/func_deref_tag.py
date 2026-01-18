from __future__ import annotations
import os
import stat
from pathlib import Path
from string import digits
from git.exc import WorkTreeRepositoryUnsupported
from git.objects import Object
from git.refs import SymbolicReference
from git.util import hex_to_bin, bin_to_hex, cygpath
from gitdb.exc import (
import os.path as osp
from git.cmd import Git
from typing import Union, Optional, cast, TYPE_CHECKING
from git.types import Commit_ish
def deref_tag(tag: 'Tag') -> 'TagObject':
    """Recursively dereference a tag and return the resulting object"""
    while True:
        try:
            tag = tag.object
        except AttributeError:
            break
    return tag