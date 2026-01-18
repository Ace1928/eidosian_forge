from .. import utils
from .._lazyload import rpy2
from . import conversion
import functools
def install_bioconductor(package=None, site_repository=None, update=False, type='binary', version=None, verbose=True):
    """Install a Bioconductor package.

    Parameters
    ----------
    site_repository : string, optional (default: None)
        additional repository in which to look for packages to install.
        This repository will be prepended to the default repositories
    update : boolean, optional (default: False)
        When False, don't attempt to update old packages.
        When True, update old packages automatically.
    type : {"binary", "source", "both"}, optional (default: "binary")
        Which package version to install if a newer version is available as source.
        "both" tries source first and uses binary as a fallback.
    version : string, optional (default: None)
        Bioconductor version to install, e.g., version = "3.8".
        The special symbol version = "devel" installs the current 'development' version.
        If None, installs from the current version.
    verbose : boolean, optional (default: True)
        Install script verbosity.
    """
    kwargs = {'update': update, 'rpy_verbose': verbose, 'type': type}
    if package is not None:
        kwargs['package'] = package
    if site_repository is not None:
        kwargs['site_repository'] = site_repository
    if version is not None:
        kwargs['version'] = version
    _install_bioconductor(**kwargs)