from __future__ import (absolute_import, division, print_function)
import os
import re
import traceback
from operator import itemgetter
def delete_all_hosts(self):
    self.config_data = []
    self.write_to_ssh_config()
    return self