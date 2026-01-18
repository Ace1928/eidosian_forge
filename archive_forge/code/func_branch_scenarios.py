from breezy import errors, tests
from breezy.branch import format_registry
from breezy.bzr.remote import RemoteBranchFormat
from breezy.tests import test_server
from breezy.tests.per_controldir.test_controldir import TestCaseWithControlDir
from breezy.transport import memory
def branch_scenarios():
    """ """
    combinations = [(format, format._matchingcontroldir) for format in format_registry._get_all()]
    scenarios = make_scenarios(None, None, combinations)
    remote_branch_format = RemoteBranchFormat()
    scenarios.extend(make_scenarios(test_server.SmartTCPServer_for_testing, test_server.ReadonlySmartTCPServer_for_testing, [(remote_branch_format, remote_branch_format._matchingcontroldir)], memory.MemoryServer, name_suffix='-default'))
    scenarios.extend(make_scenarios(test_server.SmartTCPServer_for_testing_v2_only, test_server.ReadonlySmartTCPServer_for_testing_v2_only, [(remote_branch_format, remote_branch_format._matchingcontroldir)], memory.MemoryServer, name_suffix='-v2'))
    return scenarios