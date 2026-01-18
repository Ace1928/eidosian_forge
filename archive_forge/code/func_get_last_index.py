from __future__ import (absolute_import, division, print_function)
import os
import re
import traceback
from operator import itemgetter
def get_last_index(self):
    last_index = 0
    indexes = []
    for item in self.config_data:
        if item.get('order'):
            indexes.append(item.get('order'))
    if len(indexes) > 0:
        last_index = max(indexes)
    return last_index