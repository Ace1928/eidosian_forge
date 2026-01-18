import sys
from typing import List, Optional, Set, Tuple
from pip._vendor.packaging.tags import Tag
from pip._internal.utils.compatibility_tags import get_supported, version_info_to_nodot
from pip._internal.utils.misc import normalize_version_info
def format_given(self) -> str:
    """
        Format the given, non-None attributes for display.
        """
    display_version = None
    if self._given_py_version_info is not None:
        display_version = '.'.join((str(part) for part in self._given_py_version_info))
    key_values = [('platforms', self.platforms), ('version_info', display_version), ('abis', self.abis), ('implementation', self.implementation)]
    return ' '.join((f'{key}={value!r}' for key, value in key_values if value is not None))