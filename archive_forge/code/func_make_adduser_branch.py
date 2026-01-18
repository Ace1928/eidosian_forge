import os
from breezy import merge, osutils, tests
from breezy.plugins import po_merge
from breezy.tests import features, script
def make_adduser_branch(test, relpath):
    """Helper for po_merge blackbox tests.

    This creates a branch containing the needed base revisions so tests can
    attempt merges and conflict resolutions.
    """
    builder = test.make_branch_builder(relpath)
    builder.start_series()
    builder.build_snapshot(None, [('add', ('', b'root-id', 'directory', '')), ('add', ('po', b'dir-id', 'directory', None)), ('add', ('po/adduser.pot', b'pot-id', 'file', _Adduser['base_pot'])), ('add', ('po/fr.po', b'po-id', 'file', _Adduser['base_po']))], revision_id=b'base')
    builder.build_snapshot([b'base'], [('modify', ('po/adduser.pot', _Adduser['other_pot'])), ('modify', ('po/fr.po', _Adduser['other_po']))], revision_id=b'other')
    builder.build_snapshot([b'base'], [('modify', ('po/adduser.pot', _Adduser['this_pot'])), ('modify', ('po/fr.po', _Adduser['this_po']))], revision_id=b'this')
    builder.finish_series()
    return builder