import pkgutil
import sys
from oslo_concurrency import lockutils
from oslo_log import log as logging
from oslo_utils import importutils
from stevedore import driver
from stevedore import enabled
from neutron_lib._i18n import _
def list_package_modules(package_name):
    """Get a list of the modules for a given package.

    :param package_name: The package name to get modules for.
    :returns: A list of module objects for the said package name.
    """
    pkg_mod = importutils.import_module(package_name)
    modules = [pkg_mod]
    for mod in pkgutil.walk_packages(pkg_mod.__path__):
        _, mod_name, _ = mod
        fq_name = pkg_mod.__name__ + '.' + mod_name
        modules.append(importutils.import_module(fq_name))
    return modules