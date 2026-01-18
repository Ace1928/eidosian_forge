import os
import sys
from typing import List
from pip._vendor import platformdirs as _appdirs
def site_config_dirs(appname: str) -> List[str]:
    if sys.platform == 'darwin':
        return [_appdirs.site_data_dir(appname, appauthor=False, multipath=True)]
    dirval = _appdirs.site_config_dir(appname, appauthor=False, multipath=True)
    if sys.platform == 'win32':
        return [dirval]
    return dirval.split(os.pathsep) + ['/etc']