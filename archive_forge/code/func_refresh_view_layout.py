from kivy.compat import string_types
from kivy.factory import Factory
from kivy.properties import StringProperty, ObjectProperty
from kivy.uix.behaviors import CompoundSelectionBehavior
from kivy.uix.recycleview.views import RecycleDataViewBehavior, \
def refresh_view_layout(self, index, layout, view, viewport):
    """`See :meth:`~kivy.uix.recycleview.views.RecycleDataAdapter.refresh_view_layout`.
        """
    self.recycleview.view_adapter.refresh_view_layout(index, layout, view, viewport)