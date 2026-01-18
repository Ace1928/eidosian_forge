import contextlib
import os
import re
import subprocess
import sys
import tempfile
from io import BytesIO
from .. import diff, errors, osutils
from .. import revision as _mod_revision
from .. import revisionspec, revisiontree, tests
from ..tests import EncodingAdapter, features
from ..tests.scenarios import load_tests_apply_scenarios
def test_binary_lines(self):
    empty = []
    uni_lines = [1023 * b'a' + b'\x00']
    self.assertRaises(errors.BinaryFile, udiff_lines, uni_lines, empty)
    self.assertRaises(errors.BinaryFile, udiff_lines, empty, uni_lines)
    udiff_lines(uni_lines, empty, allow_binary=True)
    udiff_lines(empty, uni_lines, allow_binary=True)