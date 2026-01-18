from kivy.compat import string_types
from kivy.factory import Factory
from kivy.properties import StringProperty, ObjectProperty
from kivy.uix.behaviors import CompoundSelectionBehavior
from kivy.uix.recycleview.views import RecycleDataViewBehavior, \
def detach_recycleview(self):
    self.clear_layout()
    rv = self.recycleview
    if rv:
        funbind = self.funbind
        funbind('viewclass', rv.refresh_from_data)
        funbind('key_viewclass', rv.refresh_from_data)
        funbind('viewclass', rv._dispatch_prop_on_source, 'viewclass')
        funbind('key_viewclass', rv._dispatch_prop_on_source, 'key_viewclass')
    self.recycleview = None