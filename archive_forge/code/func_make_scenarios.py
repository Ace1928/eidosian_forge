import contextlib
from breezy import errors, tests, transform, transport
from breezy.bzr.workingtree_4 import (DirStateRevisionTree, WorkingTreeFormat4,
from breezy.git.tree import GitRevisionTree
from breezy.git.workingtree import GitWorkingTreeFormat
from breezy.revisiontree import RevisionTree
from breezy.tests import features
from breezy.tests.per_controldir.test_controldir import TestCaseWithControlDir
from breezy.tests.per_workingtree import make_scenario as wt_make_scenario
from breezy.tests.per_workingtree import make_scenarios as wt_make_scenarios
from breezy.workingtree import format_registry
def make_scenarios(transport_server, transport_readonly_server, formats):
    """Generate test suites for each Tree implementation in breezy.

    Currently this covers all working tree formats, and RevisionTree and
    DirStateRevisionTree by committing a working tree to create the revision
    tree.
    """
    scenarios = wt_make_scenarios(transport_server, transport_readonly_server, formats)
    for scenario in scenarios:
        scenario[1]['_workingtree_to_test_tree'] = return_parameter
    workingtree_format = format_registry.get_default()
    scenarios.append((RevisionTree.__name__, create_tree_scenario(transport_server, transport_readonly_server, workingtree_format, revision_tree_from_workingtree)))
    scenarios.append((GitRevisionTree.__name__, create_tree_scenario(transport_server, transport_readonly_server, GitWorkingTreeFormat(), revision_tree_from_workingtree)))
    scenarios.append((DirStateRevisionTree.__name__ + ',WT4', create_tree_scenario(transport_server, transport_readonly_server, WorkingTreeFormat4(), _dirstate_tree_from_workingtree)))
    scenarios.append((DirStateRevisionTree.__name__ + ',WT5', create_tree_scenario(transport_server, transport_readonly_server, WorkingTreeFormat5(), _dirstate_tree_from_workingtree)))
    scenarios.append(('PreviewTree', create_tree_scenario(transport_server, transport_readonly_server, workingtree_format, preview_tree_pre)))
    scenarios.append(('PreviewTreePost', create_tree_scenario(transport_server, transport_readonly_server, workingtree_format, preview_tree_post)))
    return scenarios