import os
import re
import sys
import numpy as np
import inspect
import sysconfig
def get_max_dtype_list_length(self):
    max_len = 0
    for arg_idx in self.arguments:
        cur_len = len(self.arguments[arg_idx][1])
        if cur_len > max_len:
            max_len = cur_len
    return max_len