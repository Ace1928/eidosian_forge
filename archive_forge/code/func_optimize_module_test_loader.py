import fixtures
import os
import testresources
import testscenarios
from oslo_db import exception
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import provision
from oslo_db.sqlalchemy import utils
def optimize_module_test_loader():
    """Organize module-level tests into a testresources.OptimizingTestSuite.

    This function provides a unittest-compatible load_tests hook
    for a given module; for per-package, use the
    :func:`.optimize_package_test_loader` function.

    When a unitest or subunit style
    test runner is used, the function will be called in order to
    return a TestSuite containing the tests to run; this function
    ensures that this suite is an OptimisingTestSuite, which will organize
    the production of test resources across groups of tests at once.

    The function is invoked as::

        from oslo_db.sqlalchemy import test_fixtures

        load_tests = test_fixtures.optimize_module_test_loader()

    The loader *must* be present in an individual module, and *not* the
    package level __init__.py.

    The function also applies testscenarios expansion to all  test collections.
    This so that an existing test suite that already needs to build
    TestScenarios from a load_tests call can still have this take place when
    replaced with this function.

    """

    def load_tests(loader, found_tests, pattern):
        result = testresources.OptimisingTestSuite()
        result.addTests(testscenarios.generate_scenarios(found_tests))
        return result
    return load_tests