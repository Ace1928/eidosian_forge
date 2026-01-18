import gyp.common
import json
import os
import posixpath
def is_build_impacted(self):
    """Returns true if the supplied files impact the build at all."""
    return self._changed_targets