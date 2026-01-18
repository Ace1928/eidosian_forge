import unittest
from unittest import mock
from traits.has_traits import HasTraits
from traits.observation._dict_item_observer import DictItemObserver
from traits.observation._testing import (
from traits.trait_dict_object import TraitDict
from traits.trait_types import Dict, Str
def test_maintain_notifier_for_removed(self):
    instance = ClassWithDict(dict_of_dict={'1': {'2': 2}})
    graph = create_graph(create_observer(notify=False, optional=False), create_observer(notify=True, optional=False))
    handler = mock.Mock()
    call_add_or_remove_notifiers(object=instance.dict_of_dict, graph=graph, handler=handler)
    inner_dict = instance.dict_of_dict['1']
    inner_dict['3'] = 3
    self.assertEqual(handler.call_count, 1)
    ((event,), _), = handler.call_args_list
    self.assertEqual(event.added, {'3': 3})
    self.assertEqual(event.removed, {})
    handler.reset_mock()
    instance.dict_of_dict['1'] = {}
    inner_dict['4'] = 4
    self.assertEqual(handler.call_count, 0)