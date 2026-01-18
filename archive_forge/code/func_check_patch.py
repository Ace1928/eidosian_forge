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
def check_patch(self, lines):
    self.assertTrue(len(lines) > 1)
    self.assertTrue(lines[0].startswith(b'---'))
    self.assertTrue(lines[1].startswith(b'+++'))
    self.assertTrue(len(lines) > 2)
    self.assertTrue(lines[2].startswith(b'@@'))
    self.assertTrue(b'@@' in lines[2][2:])