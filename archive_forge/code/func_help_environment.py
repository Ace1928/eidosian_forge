import os
import os.path
import sys
import glob
import textwrap
import logging
import socket
import subprocess
import pyomo.common
from pyomo.common.collections import Bunch
import pyomo.scripting.pyomo_parser
def help_environment():
    info = Bunch()
    info.python = Bunch()
    info.python.version = '%d.%d.%d' % sys.version_info[:3]
    info.python.executable = sys.executable
    info.python.platform = sys.platform
    try:
        packages = []
        import pip
        for package in pip.get_installed_distributions():
            packages.append(Bunch(name=package.project_name, version=package.version))
        info.python.packages = packages
    except:
        pass
    info.environment = Bunch()
    path = os.environ.get('PATH', None)
    if not path is None:
        info.environment['shell path'] = path.split(os.pathsep)
    info.environment['python path'] = sys.path
    print('#')
    print('# Information About the Python and Shell Environment')
    print('#')
    print(str(info))