from __future__ import absolute_import, division, print_function
import os
import re
from ansible_collections.community.general.plugins.module_utils.module_helper import StateModuleHelper
from ansible_collections.community.general.plugins.module_utils.mh.deco import check_mode_skip
from ansible_collections.community.general.plugins.module_utils.locale_gen import locale_runner, locale_gen_runner
def fix_case(self, name):
    """locale -a might return the encoding in either lower or upper case.
        Passing through this function makes them uniform for comparisons."""
    for s, r in self.LOCALE_NORMALIZATION.items():
        name = name.replace(s, r)
    return name