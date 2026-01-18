from breezy import errors, osutils, repository
from breezy.bzr import btree_index
from breezy.bzr.remote import RemoteRepository
from breezy.bzr.tests.per_repository_chk import TestCaseWithRepositoryCHK
from breezy.bzr.versionedfile import VersionedFiles
from breezy.tests import TestNotApplicable
def reopen_repo_and_resume_write_group(self, repo):
    resume_tokens = repo.suspend_write_group()
    repo.unlock()
    reopened_repo = repo.controldir.open_repository()
    reopened_repo.lock_write()
    self.addCleanup(reopened_repo.unlock)
    reopened_repo.resume_write_group(resume_tokens)
    return reopened_repo