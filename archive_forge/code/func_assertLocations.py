from typing import List
from breezy import branch, urlutils
from breezy.tests import script
def assertLocations(self, expected_locations):
    br, _ = branch.Branch.open_containing(self.working_dir)
    if not expected_locations:
        self.assertEqual(None, br.get_parent())
    else:
        expected_pull_location = expected_locations[0]
        pull_location = urlutils.relative_url(br.base, br.get_parent())
        self.assertIsSameRealPath(expected_pull_location, pull_location)