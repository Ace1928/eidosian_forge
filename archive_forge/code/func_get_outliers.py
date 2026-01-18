import json
import shlex
import subprocess
from typing import Tuple
import torch
def get_outliers(self, weight):
    if not self.is_initialized():
        print('Outlier tracer is not initialized...')
        return None
    hvalue = self.get_hvalue(weight)
    if hvalue in self.hvalue2outlier_idx:
        return self.hvalue2outlier_idx[hvalue]
    else:
        return None