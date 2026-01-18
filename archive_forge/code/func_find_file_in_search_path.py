from __future__ import (absolute_import, division, print_function)
from abc import abstractmethod
from ansible.errors import AnsibleFileNotFound
from ansible.plugins import AnsiblePlugin
from ansible.utils.display import Display
def find_file_in_search_path(self, myvars, subdir, needle, ignore_missing=False):
    """
        Return a file (needle) in the task's expected search path.
        """
    if 'ansible_search_path' in myvars:
        paths = myvars['ansible_search_path']
    else:
        paths = [self.get_basedir(myvars)]
    result = None
    try:
        result = self._loader.path_dwim_relative_stack(paths, subdir, needle, is_role=bool('role_path' in myvars))
    except AnsibleFileNotFound:
        if not ignore_missing:
            self._display.warning("Unable to find '%s' in expected paths (use -vvvvv to see paths)" % needle)
    return result