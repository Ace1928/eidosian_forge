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
def test_check_commit_with_unparseable_time(self):
    identity_with_wrong_time = b'Igor Sysoev <igor@sysoev.ru> 18446743887488505614+42707004'
    self.assertCheckFails(Commit, self.make_commit_text(author=default_committer, committer=identity_with_wrong_time))
    self.assertCheckFails(Commit, self.make_commit_text(author=identity_with_wrong_time, committer=default_committer))