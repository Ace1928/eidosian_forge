from kivy.compat import string_types
from kivy.factory import Factory
from kivy.properties import StringProperty, ObjectProperty
from kivy.uix.behaviors import CompoundSelectionBehavior
from kivy.uix.recycleview.views import RecycleDataViewBehavior, \
def on_viewclass(self, instance, value):
    if isinstance(value, string_types):
        self.viewclass = getattr(Factory, value)