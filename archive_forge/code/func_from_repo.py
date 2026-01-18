import hashlib
import os
import tempfile
@classmethod
def from_repo(cls, repo, create=False):
    lfs_dir = os.path.join(repo.controldir, 'lfs')
    if create:
        return cls.create(lfs_dir)
    return cls(lfs_dir)