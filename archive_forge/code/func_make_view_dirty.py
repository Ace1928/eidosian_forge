from kivy.properties import ObjectProperty
from kivy.event import EventDispatcher
from collections import defaultdict
def make_view_dirty(self, view, index):
    """(internal) Used to flag this view as dirty, ready to be used for
        others. See :meth:`make_views_dirty`.
        """
    del self.views[index]
    self.dirty_views[view.__class__][index] = view