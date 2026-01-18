import itertools
import operator
import sys
def semantic_version(self):
    """Return the SemanticVersion object for this version."""
    if self._semantic is None:
        if use_importlib:
            self._semantic = self._get_version_from_importlib_metadata()
        else:
            self._semantic = self._get_version_from_pkg_resources()
    return self._semantic