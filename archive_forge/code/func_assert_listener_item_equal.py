from functools import partial
import unittest
from traits import traits_listener
from traits.api import (
def assert_listener_item_equal(test_case, item1, item2, msg=None):
    """ Assertion function for comparing two instances of ListenerItem.
    """

    def get_msg(name, msg):
        return '{name} mismatched. {msg}'.format(name=name, msg='' if msg is None else msg)
    test_case.assertEqual(item1.name, item2.name, msg=get_msg('name', msg))
    test_case.assertEqual(item1.metadata_name, item2.metadata_name, msg=get_msg('metadata_name', msg))
    test_case.assertEqual(item1.metadata_defined, item2.metadata_defined, msg=get_msg('metadata_defined', msg))
    test_case.assertEqual(item1.is_anytrait, item2.is_anytrait, msg=get_msg('is_anytrait', msg))
    test_case.assertEqual(item1.dispatch, item2.dispatch, msg=get_msg('dispatch', msg))
    test_case.assertEqual(item1.notify, item2.notify, msg=get_msg('notify', msg))
    test_case.assertEqual(item1.is_list_handler, item2.is_list_handler, msg=get_msg('is_list_handler', msg))
    test_case.assertEqual(item1.type, item2.type, msg=get_msg('type', msg))
    if item1.next is item2.next:
        pass
    else:
        test_case.assertEqual(item1.next, item2.next, msg=get_msg('next', msg))