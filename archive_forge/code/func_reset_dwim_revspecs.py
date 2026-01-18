import datetime
import time
from breezy import errors
from breezy import revision as _mod_revision
from breezy.revisionspec import (InvalidRevisionSpec, RevisionInfo,
from breezy.tests import TestCaseWithTransport
def reset_dwim_revspecs():
    RevisionSpec_dwim._possible_revspecs = original_dwim_revspecs