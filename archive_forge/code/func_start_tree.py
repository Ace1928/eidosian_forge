from . import errors
def start_tree(self, tree):
    """Start building on tree.

        :param tree: A tree to start building on. It must provide the
            MutableTree interface.
        """
    if self._tree is not None:
        raise AlreadyBuilding
    self._tree = tree
    self._tree.lock_tree_write()