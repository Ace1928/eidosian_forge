from copy import deepcopy
from kivy.uix.scrollview import ScrollView
from kivy.properties import AliasProperty
from kivy.clock import Clock
from kivy.uix.recycleview.layout import RecycleLayoutManagerBehavior, \
from kivy.uix.recycleview.views import RecycleDataAdapter
from kivy.uix.recycleview.datamodel import RecycleDataModelBehavior, \
def refresh_from_layout(self, *largs, **kwargs):
    """
        This should be called when the layout changes or needs to change. It is
        typically called when a layout parameter has changed and therefore the
        layout needs to be recomputed.
        """
    self._refresh_flags['layout'].append(kwargs)
    self._refresh_trigger()