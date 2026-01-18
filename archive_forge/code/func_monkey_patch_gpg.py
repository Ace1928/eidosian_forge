from ... import branch, gpg
from ...tests import fixtures
from . import TestCaseWithTransport
from .matchers import ContainsNoVfsCalls
def monkey_patch_gpg(self):
    """Monkey patch the gpg signing strategy to be a loopback.

        This also registers the cleanup, so that we will revert to
        the original gpg strategy when done.
        """
    self.overrideAttr(gpg, 'GPGStrategy', gpg.LoopbackGPGStrategy)