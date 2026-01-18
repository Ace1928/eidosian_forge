from kivy.compat import string_types
from kivy.factory import Factory
from kivy.properties import StringProperty, ObjectProperty
from kivy.uix.behaviors import CompoundSelectionBehavior
from kivy.uix.recycleview.views import RecycleDataViewBehavior, \
def apply_selection(self, index, view, is_selected):
    """Applies the selection to the view. This is called internally when
        a view is displayed and it needs to be shown as selected or as not
        selected.

        It is called when :meth:`select_node` or :meth:`deselect_node` is
        called or when a view needs to be refreshed. Its function is purely to
        update the view to reflect the selection state. So the function may be
        called multiple times even if the selection state may not have changed.

        If the view is a instance of
        :class:`~kivy.uix.recycleview.views.RecycleDataViewBehavior`, its
        :meth:`~kivy.uix.recycleview.views.RecycleDataViewBehavior.apply_selection` method will be called every time the view needs to refresh
        the selection state. Otherwise, the this method is responsible
        for applying the selection.

        :Parameters:

            `index`: int
                The index of the data item that is associated with the view.
            `view`: widget
                The widget that is the view of this data item.
            `is_selected`: bool
                Whether the item is selected.
        """
    viewclass = view.__class__
    if viewclass not in _view_base_cache:
        _view_base_cache[viewclass] = isinstance(view, RecycleDataViewBehavior)
    if _view_base_cache[viewclass]:
        view.apply_selection(self.recycleview, index, is_selected)