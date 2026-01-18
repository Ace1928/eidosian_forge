from __future__ import (absolute_import, division, print_function)
import re
def detect_sysctl(self):
    self.sysctl_path = self.module.get_bin_path('sysctl')