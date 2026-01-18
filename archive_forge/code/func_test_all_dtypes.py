import os
import re
import sys
import numpy as np
import inspect
import sysconfig
def test_all_dtypes(self):
    for type_idx in range(self.get_max_dtype_list_length()):
        args_array = []
        for arg_idx in self.arguments:
            new_dtype = self.get_dtype(self.arguments[arg_idx][1], type_idx)
            args_array.append(self.arguments[arg_idx][0].astype(new_dtype))
        self.pythranfunc(*args_array)