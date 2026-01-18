import re
from . import errors, osutils, transport
def set_view_info(self, current, views):
    """Set the current view and dictionary of views.

        Args:
          current: the name of the current view or None if no view is
              enabled
          views: a map from view name to list of files/directories
        """
    if current is not None and current not in views:
        raise NoSuchView(current)
    with self.tree.lock_write():
        self._current = current
        self._views = views
        self._save_view_info()