import re
from . import errors, osutils, transport
def set_view(self, view_name, view_files, make_current=True):
    """Add or update a view definition.

        Args:
          view_name: the name of the view
          view_files: the list of files/directories in the view
          make_current: make this view the current one or not
        """
    with self.tree.lock_write():
        self._load_view_info()
        self._views[view_name] = view_files
        if make_current:
            self._current = view_name
        self._save_view_info()