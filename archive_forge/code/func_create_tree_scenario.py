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
def create_tree_scenario(transport_server, transport_readonly_server, workingtree_format, converter):
    """Create a scenario for the specified converter

    :param converter: A function that converts a workingtree into the
        desired format.
    :param workingtree_format: The particular workingtree format to
        convert from.
    :return: a (name, options) tuple, where options is a dict of values
        to be used as members of the TestCase.
    """
    scenario_options = wt_make_scenario(transport_server, transport_readonly_server, workingtree_format)
    scenario_options['_workingtree_to_test_tree'] = converter
    return scenario_options