from ..push import PushResult
from .errors import GitSmartRemoteNotSupported
def import_revisions(self, revids, lossy):
    """Import a set of revisions into this git repository.

        :param revids: Revision ids of revisions to import
        :param lossy: Whether to not roundtrip bzr metadata
        """
    for i, revid in enumerate(revids):
        if self.pb:
            self.pb.update('pushing revisions', i, len(revids))
        git_commit = self.import_revision(revid, lossy)
        yield (revid, git_commit)