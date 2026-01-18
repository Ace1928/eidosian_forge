from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
import os
import sys
def reorder_sys_path(sys_path):
    """If site packages are enabled reorder them.

  Make sure bundled_python site-packages appear first in the sys.path.

  Args:
    sys_path: list current sys path

  Returns:
    modified syspath if CLOUDSDK_PYTHON_SITEPACKAGES is on, prefer bundled
    python site packages over all other. Note the returns syspath has the same
    elements but a different order.
  """
    if 'CLOUDSDK_PYTHON_SITEPACKAGES' in os.environ:
        new_path = []
        other_site_packages = []
        for path in sys_path:
            if 'site-packages' in path and 'platform/bundledpythonunix' not in path:
                other_site_packages.append(path)
            else:
                new_path.append(path)
        new_path.extend(other_site_packages)
        return new_path
    else:
        return sys_path