from ...controldir import format_registry
from ...repository import InterRepository
from ...tests import TestCaseWithTransport
from ..interrepo import InterToGitRepository
from ..mapping import BzrGitMappingExperimental, BzrGitMappingv1
def decide(x):
    return {b'refs/heads/master': (None, last_revid)}