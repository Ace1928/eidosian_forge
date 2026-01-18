from __future__ import absolute_import, print_function, unicode_literals
import pytest  # NOQA
import sys
from .roundtrip import (
def test_issue_221_add(self):
    from srsly.ruamel_yaml.comments import CommentedSeq
    a = CommentedSeq([1, 2, 3])
    a + [4, 5]