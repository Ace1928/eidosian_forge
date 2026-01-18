import logging
import os
import sys
from distutils.cmd import Command as DistutilsCommand
from distutils.command.install import SCHEME_KEYS
from distutils.command.install import install as distutils_install_command
from distutils.sysconfig import get_python_lib
from typing import Dict, List, Optional, Union, cast
from pip._internal.models.scheme import Scheme
from pip._internal.utils.compat import WINDOWS
from pip._internal.utils.virtualenv import running_under_virtualenv
from .base import get_major_minor_version
def distutils_scheme(dist_name: str, user: bool=False, home: Optional[str]=None, root: Optional[str]=None, isolated: bool=False, prefix: Optional[str]=None, *, ignore_config_files: bool=False) -> Dict[str, str]:
    """
    Return a distutils install scheme
    """
    from distutils.dist import Distribution
    dist_args: Dict[str, Union[str, List[str]]] = {'name': dist_name}
    if isolated:
        dist_args['script_args'] = ['--no-user-cfg']
    d = Distribution(dist_args)
    if not ignore_config_files:
        try:
            d.parse_config_files()
        except UnicodeDecodeError:
            paths = d.find_config_files()
            logger.warning('Ignore distutils configs in %s due to encoding errors.', ', '.join((os.path.basename(p) for p in paths)))
    obj: Optional[DistutilsCommand] = None
    obj = d.get_command_obj('install', create=True)
    assert obj is not None
    i = cast(distutils_install_command, obj)
    assert not (user and prefix), f'user={user} prefix={prefix}'
    assert not (home and prefix), f'home={home} prefix={prefix}'
    i.user = user or i.user
    if user or home:
        i.prefix = ''
    i.prefix = prefix or i.prefix
    i.home = home or i.home
    i.root = root or i.root
    i.finalize_options()
    scheme = {}
    for key in SCHEME_KEYS:
        scheme[key] = getattr(i, 'install_' + key)
    if 'install_lib' in d.get_option_dict('install'):
        scheme.update({'purelib': i.install_lib, 'platlib': i.install_lib})
    if running_under_virtualenv():
        if home:
            prefix = home
        elif user:
            prefix = i.install_userbase
        else:
            prefix = i.prefix
        scheme['headers'] = os.path.join(prefix, 'include', 'site', f'python{get_major_minor_version()}', dist_name)
        if root is not None:
            path_no_drive = os.path.splitdrive(os.path.abspath(scheme['headers']))[1]
            scheme['headers'] = os.path.join(root, path_no_drive[1:])
    return scheme