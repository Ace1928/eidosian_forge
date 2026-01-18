import unittest
from traits.api import AbstractViewElement, HasTraits, Int, TraitError
from traits.testing.optional_dependencies import requires_traitsui
def test_included_names(self):
    from traitsui.api import Group, Item, View
    item = Item('count', id='item_with_id')
    group = Group(item)
    view = View(Item('count'))

    class Model(HasTraits):
        count = Int
        my_group = group
        my_view = view
    view_elements = Model.class_trait_view_elements()
    self.assertEqual(view_elements.content, {'my_view': view, 'my_group': group, 'item_with_id': item})