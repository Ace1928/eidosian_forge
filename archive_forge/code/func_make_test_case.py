from testscenarios import multiply_scenarios
from testtools import TestCase
from testtools.matchers import (
def make_test_case(test_method_name, set_up=None, test_body=None, tear_down=None, cleanups=(), pre_set_up=None, post_tear_down=None):
    """Make a test case with the given behaviors.

    All callables are unary callables that receive this test as their argument.

    :param str test_method_name: The name of the test method.
    :param callable set_up: Implementation of setUp.
    :param callable test_body: Implementation of the actual test. Will be
        assigned to the test method.
    :param callable tear_down: Implementation of tearDown.
    :param cleanups: Iterable of callables that will be added as cleanups.
    :param callable pre_set_up: Called before the upcall to setUp().
    :param callable post_tear_down: Called after the upcall to tearDown().

    :return: A ``testtools.TestCase``.
    """
    set_up = set_up if set_up else _do_nothing
    test_body = test_body if test_body else _do_nothing
    tear_down = tear_down if tear_down else _do_nothing
    pre_set_up = pre_set_up if pre_set_up else _do_nothing
    post_tear_down = post_tear_down if post_tear_down else _do_nothing
    return _ConstructedTest(test_method_name, set_up, test_body, tear_down, cleanups, pre_set_up, post_tear_down)