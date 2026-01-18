import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def test_merged_without_child_revisions(self):
    """Test irregular layout.

        Revision ranges can produce "holes" in the depths.
        """
    self.assertReversed([('1', 2), ('2', 2), ('3', 3), ('4', 4)], [('4', 4), ('3', 3), ('2', 2), ('1', 2)])
    self.assertReversed([('1', 2), ('2', 2), ('3', 3), ('4', 4)], [('3', 3), ('4', 4), ('2', 2), ('1', 2)])