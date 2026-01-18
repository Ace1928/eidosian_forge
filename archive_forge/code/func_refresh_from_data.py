from copy import deepcopy
from kivy.uix.scrollview import ScrollView
from kivy.properties import AliasProperty
from kivy.clock import Clock
from kivy.uix.recycleview.layout import RecycleLayoutManagerBehavior, \
from kivy.uix.recycleview.views import RecycleDataAdapter
from kivy.uix.recycleview.datamodel import RecycleDataModelBehavior, \
def refresh_from_data(self, *largs, **kwargs):
    """
        This should be called when data changes. Data changes typically
        indicate that everything should be recomputed since the source data
        changed.

        This method is automatically bound to the
        :attr:`~RecycleDataModelBehavior.on_data_changed` method of the
        :class:`~RecycleDataModelBehavior` class and
        therefore responds to and accepts the keyword arguments of that event.

        It can be called manually to trigger an update.
        """
    self._refresh_flags['data'].append(kwargs)
    self._refresh_trigger()