from copy import deepcopy
from kivy.uix.scrollview import ScrollView
from kivy.properties import AliasProperty
from kivy.clock import Clock
from kivy.uix.recycleview.layout import RecycleLayoutManagerBehavior, \
from kivy.uix.recycleview.views import RecycleDataAdapter
from kivy.uix.recycleview.datamodel import RecycleDataModelBehavior, \
def get_viewport(self):
    lm = self.layout_manager
    lm_w, lm_h = lm.size
    w, h = self.size
    scroll_y = min(1, max(self.scroll_y, 0))
    scroll_x = min(1, max(self.scroll_x, 0))
    if lm_h <= h:
        bottom = 0
    else:
        above = (lm_h - h) * scroll_y
        bottom = max(0, lm_h - above - h)
    bottom = max(0, (lm_h - h) * scroll_y)
    left = max(0, (lm_w - w) * scroll_x)
    width = min(w, lm_w)
    height = min(h, lm_h)
    left, bottom = self._convert_sv_to_lm(left, bottom)
    return (left, bottom, width, height)