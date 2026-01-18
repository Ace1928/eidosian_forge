import datetime
import time
from breezy import errors
from breezy import revision as _mod_revision
from breezy.revisionspec import (InvalidRevisionSpec, RevisionInfo,
from breezy.tests import TestCaseWithTransport
def spec_in_history(spec, branch):
    """A simple helper to change a revision spec into a branch search"""
    return RevisionSpec.from_string(spec).in_history(branch)