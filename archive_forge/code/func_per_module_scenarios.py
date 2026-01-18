from itertools import (
import sys
import unittest
from testtools.testcase import clone_test_with_new_id
from testtools import iterate_tests
def per_module_scenarios(attribute_name, modules):
    """Generate scenarios for available implementation modules.

    This is typically used when there is a subsystem implemented, for
    example, in both Python and C, and we want to apply the same tests to
    both, but the C module may sometimes not be available.

    Note: if the module can't be loaded, the sys.exc_info() tuple for the
    exception raised during import of the module is used instead of the module
    object. A common idiom is to check in setUp for that and raise a skip or
    error for that case. No special helpers are supplied in testscenarios as
    yet.

    :param attribute_name: A name to be set in the scenario parameter
        dictionary (and thence onto the test instance) pointing to the 
        implementation module (or import exception) for this scenario.

    :param modules: An iterable of (short_name, module_name), where 
        the short name is something like 'python' to put in the
        scenario name, and the long name is a fully-qualified Python module
        name.
    """
    scenarios = []
    for short_name, module_name in modules:
        try:
            mod = __import__(module_name, {}, {}, [''])
        except:
            mod = sys.exc_info()
        scenarios.append((short_name, {attribute_name: mod}))
    return scenarios