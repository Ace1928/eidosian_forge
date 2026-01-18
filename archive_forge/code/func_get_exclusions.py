import os
import sys
from itertools import product, starmap
import distutils.command.install_lib as orig
def get_exclusions(self):
    """
        Return a collections.Sized collections.Container of paths to be
        excluded for single_version_externally_managed installations.
        """
    all_packages = (pkg for ns_pkg in self._get_SVEM_NSPs() for pkg in self._all_packages(ns_pkg))
    excl_specs = product(all_packages, self._gen_exclusion_paths())
    return set(starmap(self._exclude_pkg_path, excl_specs))