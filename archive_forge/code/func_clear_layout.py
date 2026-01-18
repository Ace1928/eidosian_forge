from kivy.compat import string_types
from kivy.factory import Factory
from kivy.properties import StringProperty, ObjectProperty
from kivy.uix.behaviors import CompoundSelectionBehavior
from kivy.uix.recycleview.views import RecycleDataViewBehavior, \
def clear_layout(self):
    rv = self.recycleview
    if rv:
        adapter = rv.view_adapter
        if adapter:
            adapter.invalidate()