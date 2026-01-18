import fixtures
import os
import testresources
import testscenarios
from oslo_db import exception
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import provision
from oslo_db.sqlalchemy import utils
def optimize_package_test_loader(file_):
    """Organize package-level tests into a testresources.OptimizingTestSuite.

    This function provides a unittest-compatible load_tests hook
    for a given package; for per-module, use the
    :func:`.optimize_module_test_loader` function.

    When a unitest or subunit style
    test runner is used, the function will be called in order to
    return a TestSuite containing the tests to run; this function
    ensures that this suite is an OptimisingTestSuite, which will organize
    the production of test resources across groups of tests at once.

    The function is invoked as::

        from oslo_db.sqlalchemy import test_fixtures

        load_tests = test_fixtures.optimize_package_test_loader(__file__)

    The loader *must* be present in the package level __init__.py.

    The function also applies testscenarios expansion to all  test collections.
    This so that an existing test suite that already needs to build
    TestScenarios from a load_tests call can still have this take place when
    replaced with this function.

    """
    this_dir = os.path.dirname(file_)

    def load_tests(loader, found_tests, pattern):
        result = testresources.OptimisingTestSuite()
        result.addTests(found_tests)
        pkg_tests = loader.discover(start_dir=this_dir, pattern=pattern)
        result.addTests(testscenarios.generate_scenarios(pkg_tests))
        return result
    return load_tests