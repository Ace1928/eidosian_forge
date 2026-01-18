import configparser
import glob
import os
import sys
from os.path import join as pjoin
from packaging.version import Version
from .environment import get_nipy_system_dir, get_nipy_user_dir
def datasource_or_bomber(pkg_def, **options):
    """Return a viable datasource or a Bomber

    This is to allow module level creation of datasource objects.  We
    create the objects, so that, if the data exist, and are the correct
    version, the objects are valid datasources, otherwise, they
    raise an error on access, warning about the lack of data or the
    version numbers.

    The parameters are as for ``make_datasource`` in this module.

    Parameters
    ----------
    pkg_def : dict
       dict containing at least key 'relpath'. Can optionally have keys 'name'
       (package name),  'install hint' (for helpful error messages) and 'min
       version' giving the minimum necessary version string for the package.
    data_path : sequence of strings or None, optional

    Returns
    -------
    ds : datasource or ``Bomber`` instance
    """
    unix_relpath = pkg_def['relpath']
    version = pkg_def.get('min version')
    pkg_hint = pkg_def.get('install hint', DEFAULT_INSTALL_HINT)
    names = unix_relpath.split('/')
    sys_relpath = os.path.sep.join(names)
    try:
        ds = make_datasource(pkg_def, **options)
    except DataError as e:
        return Bomber(sys_relpath, str(e))
    if version is None or Version(ds.version) >= Version(version):
        return ds
    if 'name' in pkg_def:
        pkg_name = pkg_def['name']
    else:
        pkg_name = 'data at ' + unix_relpath
    msg = f'{pkg_name} is version {ds.version} but we need version >= {version}\n\n{pkg_hint}'
    return Bomber(sys_relpath, DataError(msg))