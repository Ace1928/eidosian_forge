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
def assertReadableByAttrib(self, cwd, relpath, regex):
    proc = subprocess.Popen(['attrib', relpath], stdout=subprocess.PIPE, cwd=cwd)
    result, err = proc.communicate()
    self.assertContainsRe(result.replace('\r\n', '\n'), regex)