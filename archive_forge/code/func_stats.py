import datetime
import re
from subprocess import Popen, PIPE
from gitdb import IStream
from git.util import hex_to_bin, Actor, Stats, finalize_process
from git.diff import Diffable
from git.cmd import Git
from .tree import Tree
from . import base
from .util import (
from time import time, daylight, altzone, timezone, localtime
import os
from io import BytesIO
import logging
from collections import defaultdict
from typing import (
from git.types import PathLike, Literal
@property
def stats(self) -> Stats:
    """Create a git stat from changes between this commit and its first parent
        or from all changes done if this is the very first commit.

        :return: git.Stats
        """
    if not self.parents:
        text = self.repo.git.diff_tree(self.hexsha, '--', numstat=True, no_renames=True, root=True)
        text2 = ''
        for line in text.splitlines()[1:]:
            insertions, deletions, filename = line.split('\t')
            text2 += '%s\t%s\t%s\n' % (insertions, deletions, filename)
        text = text2
    else:
        text = self.repo.git.diff(self.parents[0].hexsha, self.hexsha, '--', numstat=True, no_renames=True)
    return Stats._list_from_string(self.repo, text)