import sys, os
import textwrap
def get_option_group(self, opt_str):
    option = self._short_opt.get(opt_str) or self._long_opt.get(opt_str)
    if option and option.container is not self:
        return option.container
    return None