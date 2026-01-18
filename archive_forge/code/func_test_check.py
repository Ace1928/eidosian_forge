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
def test_check(self):
    self.assertCheckSucceeds(Tag, self.make_tag_text())
    self.assertCheckFails(Tag, self.make_tag_text(object_sha=None))
    self.assertCheckFails(Tag, self.make_tag_text(object_type_name=None))
    self.assertCheckFails(Tag, self.make_tag_text(name=None))
    self.assertCheckFails(Tag, self.make_tag_text(name=b''))
    self.assertCheckFails(Tag, self.make_tag_text(object_type_name=b'foobar'))
    self.assertCheckFails(Tag, self.make_tag_text(tagger=b'some guy without an email address 1183319674 -0700'))
    self.assertCheckFails(Tag, self.make_tag_text(tagger=b'Linus Torvalds <torvalds@woody.linux-foundation.org> Sun 7 Jul 2007 12:54:34 +0700'))
    self.assertCheckFails(Tag, self.make_tag_text(object_sha=b'xxx'))