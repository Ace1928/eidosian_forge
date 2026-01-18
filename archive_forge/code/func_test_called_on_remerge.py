import os
from breezy import merge, osutils, tests
from breezy.plugins import po_merge
from breezy.tests import features, script
def test_called_on_remerge(self):
    self.run_script('$ brz branch adduser -rrevid:this work\n2>Branched 2 revisions.\n$ cd work\n# set po_dirs to an empty list\n$ brz merge ../adduser -rrevid:other -Opo_merge.po_dirs=\n2> M  po/adduser.pot\n2> M  po/fr.po\n2>Text conflict in po/adduser.pot\n2>Text conflict in po/fr.po\n2>2 conflicts encountered.\n')
    with open('po/adduser.pot', 'wb') as f:
        f.write(_Adduser['resolved_pot'])
    self.run_script('$ brz resolve po/adduser.pot\n2>1 conflict resolved, 1 remaining\n# Use remerge to trigger the hook, we use the default config options here\n$ brz remerge po/*.po\n2>All changes applied successfully.\n# There should be no conflicts anymore\n$ brz conflicts\n')