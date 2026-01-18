import os
import platform
import subprocess
import sys
from collections import OrderedDict
from typing import Callable, Dict, Optional, Tuple, Union
from tqdm.auto import tqdm
from cmdstanpy import _DOT_CMDSTAN
from .. import progress as progbar
from .logging import get_logger
def cxx_toolchain_path(version: Optional[str]=None, install_dir: Optional[str]=None) -> Tuple[str, ...]:
    """
    Validate, then activate C++ toolchain directory path.
    """
    if platform.system() != 'Windows':
        raise RuntimeError('Functionality is currently only supported on Windows')
    if version is not None and (not isinstance(version, str)):
        raise TypeError('Format version number as a string')
    logger = get_logger()
    if 'CMDSTAN_TOOLCHAIN' in os.environ:
        toolchain_root = os.environ['CMDSTAN_TOOLCHAIN']
        if os.path.exists(os.path.join(toolchain_root, 'mingw64')):
            compiler_path = os.path.join(toolchain_root, 'mingw64' if sys.maxsize > 2 ** 32 else 'mingw32', 'bin')
            if os.path.exists(compiler_path):
                tool_path = os.path.join(toolchain_root, 'usr', 'bin')
                if not os.path.exists(tool_path):
                    tool_path = ''
                    compiler_path = ''
                    logger.warning('Found invalid installion for RTools40 on %s', toolchain_root)
                    toolchain_root = ''
            else:
                compiler_path = ''
                logger.warning('Found invalid installion for RTools40 on %s', toolchain_root)
                toolchain_root = ''
        elif os.path.exists(os.path.join(toolchain_root, 'mingw_64')):
            compiler_path = os.path.join(toolchain_root, 'mingw_64' if sys.maxsize > 2 ** 32 else 'mingw_32', 'bin')
            if os.path.exists(compiler_path):
                tool_path = os.path.join(toolchain_root, 'bin')
                if not os.path.exists(tool_path):
                    tool_path = ''
                    compiler_path = ''
                    logger.warning('Found invalid installion for RTools35 on %s', toolchain_root)
                    toolchain_root = ''
            else:
                compiler_path = ''
                logger.warning('Found invalid installion for RTools35 on %s', toolchain_root)
                toolchain_root = ''
    else:
        rtools40_home = os.environ.get('RTOOLS40_HOME')
        cmdstan_dir = os.path.expanduser(os.path.join('~', _DOT_CMDSTAN))
        for toolchain_root in ([rtools40_home] if rtools40_home is not None else []) + ([os.path.join(install_dir, 'RTools40'), os.path.join(install_dir, 'RTools35'), os.path.join(install_dir, 'RTools30'), os.path.join(install_dir, 'RTools')] if install_dir is not None else []) + [os.path.join(cmdstan_dir, 'RTools40'), os.path.join(os.path.abspath('/'), 'RTools40'), os.path.join(cmdstan_dir, 'RTools35'), os.path.join(os.path.abspath('/'), 'RTools35'), os.path.join(cmdstan_dir, 'RTools'), os.path.join(os.path.abspath('/'), 'RTools'), os.path.join(os.path.abspath('/'), 'RBuildTools')]:
            compiler_path = ''
            tool_path = ''
            if os.path.exists(toolchain_root):
                if version not in ('35', '3.5', '3'):
                    compiler_path = os.path.join(toolchain_root, 'mingw64' if sys.maxsize > 2 ** 32 else 'mingw32', 'bin')
                    if os.path.exists(compiler_path):
                        tool_path = os.path.join(toolchain_root, 'usr', 'bin')
                        if not os.path.exists(tool_path):
                            tool_path = ''
                            compiler_path = ''
                            logger.warning('Found invalid installation for RTools40 on %s', toolchain_root)
                            toolchain_root = ''
                        else:
                            break
                    else:
                        compiler_path = ''
                        logger.warning('Found invalid installation for RTools40 on %s', toolchain_root)
                        toolchain_root = ''
                else:
                    compiler_path = os.path.join(toolchain_root, 'mingw_64' if sys.maxsize > 2 ** 32 else 'mingw_32', 'bin')
                    if os.path.exists(compiler_path):
                        tool_path = os.path.join(toolchain_root, 'bin')
                        if not os.path.exists(tool_path):
                            tool_path = ''
                            compiler_path = ''
                            logger.warning('Found invalid installation for RTools35 on %s', toolchain_root)
                            toolchain_root = ''
                        else:
                            break
                    else:
                        compiler_path = ''
                        logger.warning('Found invalid installation for RTools35 on %s', toolchain_root)
                        toolchain_root = ''
            else:
                toolchain_root = ''
    if not toolchain_root:
        raise ValueError('no RTools toolchain installation found, run command line script "python -m cmdstanpy.install_cxx_toolchain"')
    logger.info('Add C++ toolchain to $PATH: %s', toolchain_root)
    os.environ['PATH'] = ';'.join(list(OrderedDict.fromkeys([compiler_path, tool_path] + os.getenv('PATH', '').split(';'))))
    return (compiler_path, tool_path)