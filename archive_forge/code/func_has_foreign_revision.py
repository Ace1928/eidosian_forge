from . import errors, registry
from .branch import Branch
from .repository import Repository
from .revision import Revision
def has_foreign_revision(self, foreign_revid):
    """Check whether the specified foreign revision is present.

        :param foreign_revid: A foreign revision id, in the format used
                              by this Repository's VCS.
        """
    raise NotImplementedError(self.has_foreign_revision)