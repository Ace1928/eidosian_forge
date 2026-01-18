from . import errors
def finish_tree(self):
    """Finish building the current tree."""
    self._ensure_building()
    tree = self._tree
    self._tree = None
    tree.unlock()