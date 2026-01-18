from breezy import errors, urlutils
from breezy.bzr import remote
from breezy.controldir import ControlDir
from breezy.tests import multiply_tests
from breezy.tests.per_repository import (TestCaseWithRepository,
def external_reference_test_scenarios():
    """Generate test scenarios for repositories supporting external references.
    """
    result = []
    for test_name, scenario_info in all_repository_format_scenarios():
        format = scenario_info['repository_format']
        if isinstance(format, remote.RemoteRepositoryFormat) or format.supports_external_lookups:
            result.append((test_name, scenario_info))
    return result