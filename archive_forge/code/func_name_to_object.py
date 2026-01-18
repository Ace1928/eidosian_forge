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
def name_to_object(repo: 'Repo', name: str, return_ref: bool=False) -> Union[SymbolicReference, 'Commit', 'TagObject', 'Blob', 'Tree']:
    """
    :return: object specified by the given name, hexshas ( short and long )
        as well as references are supported
    :param return_ref: if name specifies a reference, we will return the reference
        instead of the object. Otherwise it will raise BadObject or BadName
    """
    hexsha: Union[None, str, bytes] = None
    if repo.re_hexsha_shortened.match(name):
        if len(name) != 40:
            hexsha = short_to_long(repo.odb, name)
        else:
            hexsha = name
    if hexsha is None:
        for base in ('%s', 'refs/%s', 'refs/tags/%s', 'refs/heads/%s', 'refs/remotes/%s', 'refs/remotes/%s/HEAD'):
            try:
                hexsha = SymbolicReference.dereference_recursive(repo, base % name)
                if return_ref:
                    return SymbolicReference(repo, base % name)
                break
            except ValueError:
                pass
    if return_ref:
        raise BadObject("Couldn't find reference named %r" % name)
    if hexsha is None:
        raise BadName(name)
    return Object.new_from_sha(repo, hex_to_bin(hexsha))