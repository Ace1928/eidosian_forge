import os
import re
import sys
from typing import List, Optional
from pip._internal.locations import site_packages, user_site
from pip._internal.utils.virtualenv import (
def egg_link_path_from_sys_path(raw_name: str) -> Optional[str]:
    """
    Look for a .egg-link file for project name, by walking sys.path.
    """
    egg_link_names = _egg_link_names(raw_name)
    for path_item in sys.path:
        for egg_link_name in egg_link_names:
            egg_link = os.path.join(path_item, egg_link_name)
            if os.path.isfile(egg_link):
                return egg_link
    return None