import datetime
import os
import stat
from contextlib import contextmanager
from io import BytesIO
from itertools import permutations
from dulwich.tests import TestCase
from ..errors import ObjectFormatException
from ..objects import (
from .utils import ext_functest_builder, functest_builder, make_commit, make_object
def make_commit_lines(self, tree=b'd80c186a03f423a81b39df39dc87fd269736ca86', parents=[b'ab64bbdcc51b170d21588e5c5d391ee5c0c96dfd', b'4cffe90e0a41ad3f5190079d7c8f036bde29cbe6'], author=default_committer, committer=default_committer, encoding=None, message=b'Merge ../b\n', extra=None):
    lines = []
    if tree is not None:
        lines.append(b'tree ' + tree)
    if parents is not None:
        lines.extend((b'parent ' + p for p in parents))
    if author is not None:
        lines.append(b'author ' + author)
    if committer is not None:
        lines.append(b'committer ' + committer)
    if encoding is not None:
        lines.append(b'encoding ' + encoding)
    if extra is not None:
        for name, value in sorted(extra.items()):
            lines.append(name + b' ' + value)
    lines.append(b'')
    if message is not None:
        lines.append(message)
    return lines