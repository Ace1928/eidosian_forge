from __future__ import absolute_import, division, print_function
from ansible_collections.community.routeros.plugins.module_utils.version import LooseVersion
def provide_version(self, version):
    if not self.needs_version:
        return (self.unversioned.fully_understood, None)
    api_version = LooseVersion(version)
    if self.unversioned is not None:
        self._current = self.unversioned.specialize_for_version(api_version)
        return (self._current.fully_understood, None)
    for other_version, comparator, data in self.versioned:
        if other_version == '*' and comparator == '*':
            return self._select(data, api_version)
        other_api_version = LooseVersion(other_version)
        if _compare(api_version, other_api_version, comparator):
            return self._select(data, api_version)
    self._current = None
    return (False, None)