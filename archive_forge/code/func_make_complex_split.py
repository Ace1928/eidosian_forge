from breezy import errors
from breezy.bzr import knit
from breezy.tests.per_repository_reference import \
def make_complex_split(self):
    """intermix the revisions so that base holds left stacked holds right.

        base will hold
            A B D F (and C because it is a parent of D)
        referring will hold
            C E G (only)
        """
    self.base_repo.fetch(self.all_repo, revision_id=b'B')
    self.stacked_repo.fetch(self.all_repo, revision_id=b'C')
    self.base_repo.fetch(self.all_repo, revision_id=b'F')
    self.stacked_repo.fetch(self.all_repo, revision_id=b'G')