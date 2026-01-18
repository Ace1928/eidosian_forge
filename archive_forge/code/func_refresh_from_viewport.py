from copy import deepcopy
from kivy.uix.scrollview import ScrollView
from kivy.properties import AliasProperty
from kivy.clock import Clock
from kivy.uix.recycleview.layout import RecycleLayoutManagerBehavior, \
from kivy.uix.recycleview.views import RecycleDataAdapter
from kivy.uix.recycleview.datamodel import RecycleDataModelBehavior, \
def refresh_from_viewport(self, *largs):
    """
        This should be called when the viewport changes and the displayed data
        must be updated. Neither the data nor the layout will be recomputed.
        """
    self._refresh_flags['viewport'] = True
    self._refresh_trigger()