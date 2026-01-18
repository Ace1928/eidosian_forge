from copy import deepcopy
from kivy.uix.scrollview import ScrollView
from kivy.properties import AliasProperty
from kivy.clock import Clock
from kivy.uix.recycleview.layout import RecycleLayoutManagerBehavior, \
from kivy.uix.recycleview.views import RecycleDataAdapter
from kivy.uix.recycleview.datamodel import RecycleDataModelBehavior, \
def refresh_views(self, *largs):
    lm = self.layout_manager
    flags = self._refresh_flags
    if lm is None or self.view_adapter is None or self.data_model is None:
        return
    data = self.data
    f = flags['data']
    if f:
        self.save_viewport()
        flags['data'] = []
        flags['layout'] = [{}]
        lm.compute_sizes_from_data(data, f)
    while flags['layout']:
        self.save_viewport()
        if flags['data']:
            return
        flags['viewport'] = True
        f = flags['layout']
        flags['layout'] = []
        try:
            lm.compute_layout(data, f)
        except LayoutChangeException:
            flags['layout'].append({})
            continue
    if flags['data']:
        return
    self._refresh_trigger.cancel()
    self.restore_viewport()
    if flags['viewport']:
        flags['viewport'] = False
        viewport = self.get_viewport()
        indices = lm.compute_visible_views(data, viewport)
        lm.set_visible_views(indices, data, viewport)