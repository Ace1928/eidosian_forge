import configparser
import locale
import os
import sys
from typing import Any, Dict, Iterable, List, NewType, Optional, Tuple
from pip._internal.exceptions import (
from pip._internal.utils import appdirs
from pip._internal.utils.compat import WINDOWS
from pip._internal.utils.logging import getLogger
from pip._internal.utils.misc import ensure_dir, enum
def get_configuration_files() -> Dict[Kind, List[str]]:
    global_config_files = [os.path.join(path, CONFIG_BASENAME) for path in appdirs.site_config_dirs('pip')]
    site_config_file = os.path.join(sys.prefix, CONFIG_BASENAME)
    legacy_config_file = os.path.join(os.path.expanduser('~'), 'pip' if WINDOWS else '.pip', CONFIG_BASENAME)
    new_config_file = os.path.join(appdirs.user_config_dir('pip'), CONFIG_BASENAME)
    return {kinds.GLOBAL: global_config_files, kinds.SITE: [site_config_file], kinds.USER: [legacy_config_file, new_config_file]}