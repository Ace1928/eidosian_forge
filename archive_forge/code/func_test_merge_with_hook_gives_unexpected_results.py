import os
from breezy import merge, osutils, tests
from breezy.plugins import po_merge
from breezy.tests import features, script
def test_merge_with_hook_gives_unexpected_results(self):
    self.run_script('$ brz branch adduser -rrevid:this work\n2>Branched 2 revisions.\n$ cd work\n$ brz merge ../adduser -rrevid:other\n2> M  po/adduser.pot\n2> M  po/fr.po\n2>Text conflict in po/adduser.pot\n2>1 conflicts encountered.\n')