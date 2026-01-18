from kivy.compat import string_types
from kivy.factory import Factory
from kivy.properties import StringProperty, ObjectProperty
from kivy.uix.behaviors import CompoundSelectionBehavior
from kivy.uix.recycleview.views import RecycleDataViewBehavior, \
def attach_recycleview(self, rv):
    self.recycleview = rv
    if rv:
        fbind = self.fbind
        fbind('viewclass', rv.refresh_from_data)
        fbind('key_viewclass', rv.refresh_from_data)
        fbind('viewclass', rv._dispatch_prop_on_source, 'viewclass')
        fbind('key_viewclass', rv._dispatch_prop_on_source, 'key_viewclass')