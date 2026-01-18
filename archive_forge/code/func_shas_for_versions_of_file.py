from breezy import osutils
from breezy.bzr.inventory import Inventory, InventoryFile
from breezy.bzr.tests.per_repository_vf import (
from breezy.repository import WriteGroup
from breezy.revision import NULL_REVISION, Revision
from breezy.tests import TestNotApplicable, multiply_scenarios
from breezy.tests.scenarios import load_tests_apply_scenarios
def shas_for_versions_of_file(self, repo, versions):
    """Get the SHA-1 hashes of the versions of 'a-file' in the repository.

        :param repo: the repository to get the hashes from.
        :param versions: a list of versions to get hashes for.

        :returns: A dict of `{version: hash}`.
        """
    keys = [(b'a-file-id', version) for version in versions]
    return repo.texts.get_sha1s(keys)