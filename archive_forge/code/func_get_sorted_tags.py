import sys
from typing import List, Optional, Set, Tuple
from pip._vendor.packaging.tags import Tag
from pip._internal.utils.compatibility_tags import get_supported, version_info_to_nodot
from pip._internal.utils.misc import normalize_version_info
def get_sorted_tags(self) -> List[Tag]:
    """
        Return the supported PEP 425 tags to check wheel candidates against.

        The tags are returned in order of preference (most preferred first).
        """
    if self._valid_tags is None:
        py_version_info = self._given_py_version_info
        if py_version_info is None:
            version = None
        else:
            version = version_info_to_nodot(py_version_info)
        tags = get_supported(version=version, platforms=self.platforms, abis=self.abis, impl=self.implementation)
        self._valid_tags = tags
    return self._valid_tags