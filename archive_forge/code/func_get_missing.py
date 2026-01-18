import sys
from breezy import branch, controldir, errors, memorytree, tests
from breezy.bzr import branch as bzrbranch
from breezy.bzr import remote, versionedfile
from breezy.bzr.tests.per_repository_vf import (
from breezy.tests.scenarios import load_tests_apply_scenarios
def get_missing(check_for_missing_texts=True):
    call_log.append(check_for_missing_texts)
    return orig(check_for_missing_texts=check_for_missing_texts)