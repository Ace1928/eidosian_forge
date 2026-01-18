import gc
from io import BytesIO
import logging
import os
import os.path as osp
import stat
import uuid
import git
from git.cmd import Git
from git.compat import defenc
from git.config import GitConfigParser, SectionConstraint, cp
from git.exc import (
from git.objects.base import IndexObject, Object
from git.objects.util import TraversableIterableObj
from git.util import (
from .util import (
from typing import Callable, Dict, Mapping, Sequence, TYPE_CHECKING, cast
from typing import Any, Iterator, Union
from git.types import Commit_ish, Literal, PathLike, TBD
@classmethod
def iter_items(cls, repo: 'Repo', parent_commit: Union[Commit_ish, str]='HEAD', *Args: Any, **kwargs: Any) -> Iterator['Submodule']:
    """:return: Iterator yielding Submodule instances available in the given repository"""
    try:
        pc = repo.commit(parent_commit)
        parser = cls._config_parser(repo, pc, read_only=True)
    except (IOError, BadName):
        return
    for sms in parser.sections():
        n = sm_name(sms)
        p = parser.get(sms, 'path')
        u = parser.get(sms, 'url')
        b = cls.k_head_default
        if parser.has_option(sms, cls.k_head_option):
            b = str(parser.get(sms, cls.k_head_option))
        index = repo.index
        try:
            rt = pc.tree
            sm = rt[p]
        except KeyError:
            try:
                entry = index.entries[index.entry_key(p, 0)]
                sm = Submodule(repo, entry.binsha, entry.mode, entry.path)
            except KeyError:
                continue
        if type(sm) is not git.objects.submodule.base.Submodule:
            continue
        sm._name = n
        if pc != repo.commit():
            sm._parent_commit = pc
        sm._branch_path = git.Head.to_full_path(b)
        sm._url = u
        yield sm