from __future__ import absolute_import, division, print_function
import platform
from ansible.module_utils import distro
from ansible.module_utils.common._utils import get_all_subclasses
def get_distribution_version():
    """
    Get the version of the distribution the code is running on

    :rtype: NativeString or None
    :returns: A string representation of the version of the distribution. If it
    cannot determine the version, it returns an empty string. If this is not run on
    a Linux machine it returns None.
    """
    version = None
    needs_best_version = frozenset((u'centos', u'debian'))
    version = distro.version()
    distro_id = distro.id()
    if version is not None:
        if distro_id in needs_best_version:
            version_best = distro.version(best=True)
            if distro_id == u'centos':
                version = u'.'.join(version_best.split(u'.')[:2])
            if distro_id == u'debian':
                version = version_best
    else:
        version = u''
    return version