import os
import sys
from types import ModuleType
from .version import version as __version__  # NOQA:F401
def distribution_version(name):
    """try to get the version of the named distribution,
    returs None on failure"""
    from pkg_resources import get_distribution, DistributionNotFound
    try:
        dist = get_distribution(name)
    except DistributionNotFound:
        pass
    else:
        return dist.version