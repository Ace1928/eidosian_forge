from __future__ import absolute_import, division, print_function
import os
import re
from ansible_collections.community.general.plugins.module_utils.module_helper import StateModuleHelper
from ansible_collections.community.general.plugins.module_utils.mh.deco import check_mode_skip
from ansible_collections.community.general.plugins.module_utils.locale_gen import locale_runner, locale_gen_runner
def apply_change(self, targetState, name):
    """Create or remove locale.

        Keyword arguments:
        targetState -- Desired state, either present or absent.
        name -- Name including encoding such as de_CH.UTF-8.
        """
    self.set_locale(name, enabled=targetState == 'present')
    runner = locale_gen_runner(self.module)
    with runner() as ctx:
        ctx.run()