import logging
import shutil
import sys
from pathlib import Path
from . import EXE_FORMAT
from .config import Config
from .python_helper import PythonExecutable
def install_python_dependencies(config: Config, python: PythonExecutable, init: bool, packages: str, is_android: bool=False):
    """
        Installs the python package dependencies for the target deployment platform
    """
    packages = config.get_value('python', packages).split(',')
    if not init:
        logging.info('[DEPLOY] Installing dependencies')
        python.install(packages=packages)
        if sys.platform.startswith('linux') and (not is_android):
            python.install(packages=['patchelf'])
    elif is_android:
        logging.info('[DEPLOY] Installing buildozer')
        buildozer_package_with_version = [package for package in packages if package.startswith('buildozer')]
        python.install(packages=list(buildozer_package_with_version))