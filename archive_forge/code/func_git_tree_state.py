from pprint import pformat
from six import iteritems
import re
@git_tree_state.setter
def git_tree_state(self, git_tree_state):
    """
        Sets the git_tree_state of this VersionInfo.

        :param git_tree_state: The git_tree_state of this VersionInfo.
        :type: str
        """
    if git_tree_state is None:
        raise ValueError('Invalid value for `git_tree_state`, must not be `None`')
    self._git_tree_state = git_tree_state