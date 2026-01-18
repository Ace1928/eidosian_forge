from __future__ import annotations
import gc
import logging
import os
import os.path as osp
from pathlib import Path
import re
import shlex
import warnings
import gitdb
from gitdb.db.loose import LooseObjectDB
from gitdb.exc import BadObject
from git.cmd import Git, handle_process_output
from git.compat import defenc, safe_decode
from git.config import GitConfigParser
from git.db import GitCmdObjectDB
from git.exc import (
from git.index import IndexFile
from git.objects import Submodule, RootModule, Commit
from git.refs import HEAD, Head, Reference, TagReference
from git.remote import Remote, add_progress, to_progress_instance
from git.util import (
from .fun import (
from git.types import (
from typing import (
from git.types import ConfigLevels_Tup, TypedDict
def blame_incremental(self, rev: str | HEAD, file: str, **kwargs: Any) -> Iterator['BlameEntry']:
    """Iterator for blame information for the given file at the given revision.

        Unlike :meth:`blame`, this does not return the actual file's contents, only a
        stream of :class:`BlameEntry` tuples.

        :param rev: Revision specifier, see git-rev-parse for viable options.

        :return: Lazy iterator of :class:`BlameEntry` tuples, where the commit indicates
            the commit to blame for the line, and range indicates a span of line numbers
            in the resulting file.

        If you combine all line number ranges outputted by this command, you should get
        a continuous range spanning all line numbers in the file.
        """
    data: bytes = self.git.blame(rev, '--', file, p=True, incremental=True, stdout_as_string=False, **kwargs)
    commits: Dict[bytes, Commit] = {}
    stream = (line for line in data.split(b'\n') if line)
    while True:
        try:
            line = next(stream)
        except StopIteration:
            return
        split_line = line.split()
        hexsha, orig_lineno_b, lineno_b, num_lines_b = split_line
        lineno = int(lineno_b)
        num_lines = int(num_lines_b)
        orig_lineno = int(orig_lineno_b)
        if hexsha not in commits:
            props: Dict[bytes, bytes] = {}
            while True:
                try:
                    line = next(stream)
                except StopIteration:
                    return
                if line == b'boundary':
                    continue
                tag, value = line.split(b' ', 1)
                props[tag] = value
                if tag == b'filename':
                    orig_filename = value
                    break
            c = Commit(self, hex_to_bin(hexsha), author=Actor(safe_decode(props[b'author']), safe_decode(props[b'author-mail'].lstrip(b'<').rstrip(b'>'))), authored_date=int(props[b'author-time']), committer=Actor(safe_decode(props[b'committer']), safe_decode(props[b'committer-mail'].lstrip(b'<').rstrip(b'>'))), committed_date=int(props[b'committer-time']))
            commits[hexsha] = c
        else:
            while True:
                try:
                    line = next(stream)
                except StopIteration:
                    return
                tag, value = line.split(b' ', 1)
                if tag == b'filename':
                    orig_filename = value
                    break
        yield BlameEntry(commits[hexsha], range(lineno, lineno + num_lines), safe_decode(orig_filename), range(orig_lineno, orig_lineno + num_lines))