from collections import deque
from . import errors, revision
def get_parent_keys(self, key):
    """Get the parents for a key

        Returns a list containg the parents keys. If the key is a ghost,
        None is returned. A KeyError will be raised if the key is not in
        the graph.

        :param keys: Key to check (eg revision_id)
        :return: A list of parents
        """
    return self._nodes[key].parent_keys