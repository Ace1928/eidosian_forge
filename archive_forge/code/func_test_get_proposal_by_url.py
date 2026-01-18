import os
from typing import List
from .. import forge as _mod_forge
from .. import registry, tests, urlutils
from ..forge import (Forge, MergeProposal, UnsupportedForge, determine_title,
def test_get_proposal_by_url(self):
    self.assertRaises(UnsupportedForge, get_proposal_by_url, 'blah')
    url = urlutils.local_path_to_url(os.path.join(self.test_dir, 'hosted', 'proposal'))
    self.assertIsInstance(get_proposal_by_url(url), MergeProposal)