from kivy.properties import ObjectProperty
from kivy.event import EventDispatcher
from collections import defaultdict
def refresh_view_attrs(self, index, data_item, view):
    """(internal) Syncs the view and brings it up to date with the data.

        This method calls :meth:`RecycleDataViewBehavior.refresh_view_attrs`
        if the view inherits from :class:`RecycleDataViewBehavior`. See that
        method for more details.

        .. note::
            Any sizing and position info is skipped when syncing with the data.
        """
    viewclass = view.__class__
    if viewclass not in _view_base_cache:
        _view_base_cache[viewclass] = isinstance(view, RecycleDataViewBehavior)
    if _view_base_cache[viewclass]:
        view.refresh_view_attrs(self.recycleview, index, data_item)
    else:
        sizing_attrs = RecycleDataAdapter._sizing_attrs
        for key, value in data_item.items():
            if key not in sizing_attrs:
                setattr(view, key, value)