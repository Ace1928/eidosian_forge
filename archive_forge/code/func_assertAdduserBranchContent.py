import os
from breezy import merge, osutils, tests
from breezy.plugins import po_merge
from breezy.tests import features, script
def assertAdduserBranchContent(self, revid):
    env = dict(revid=revid, branch_name=revid)
    self.run_script('$ brz branch adduser -rrevid:%(revid)s %(branch_name)s\n' % env, null_output_matches_anything=True)
    self.assertFileEqual(_Adduser['%(revid)s_pot' % env], '%(branch_name)s/po/adduser.pot' % env)
    self.assertFileEqual(_Adduser['%(revid)s_po' % env], '%(branch_name)s/po/fr.po' % env)