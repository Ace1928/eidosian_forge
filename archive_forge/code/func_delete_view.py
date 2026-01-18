import re
from . import errors, osutils, transport
def delete_view(self, view_name):
    """Delete a view definition.

        If the view deleted is the current one, the current view is reset.
        """
    with self.tree.lock_write():
        self._load_view_info()
        try:
            del self._views[view_name]
        except KeyError:
            raise NoSuchView(view_name)
        if view_name == self._current:
            self._current = None
        self._save_view_info()