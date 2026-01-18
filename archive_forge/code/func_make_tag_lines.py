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
def make_tag_lines(self, object_sha=b'a38d6181ff27824c79fc7df825164a212eff6a3f', object_type_name=b'commit', name=b'v2.6.22-rc7', tagger=default_tagger, message=default_message):
    lines = []
    if object_sha is not None:
        lines.append(b'object ' + object_sha)
    if object_type_name is not None:
        lines.append(b'type ' + object_type_name)
    if name is not None:
        lines.append(b'tag ' + name)
    if tagger is not None:
        lines.append(b'tagger ' + tagger)
    if message is not None:
        lines.append(b'')
        lines.append(message)
    return lines