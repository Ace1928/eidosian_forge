from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
def feature_available(self, feature_version: str) -> bool:
    """Check whether the current version has the feature available.

    Args:
      feature_version: The lowest version that the feature is available.

    Returns:
      bool
    """
    return not self < Version(feature_version)