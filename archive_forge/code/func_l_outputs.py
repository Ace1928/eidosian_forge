import os
import re as regex
from ..base import (
def l_outputs(self):
    outputs = self.output_spec().get()
    for key in outputs:
        name = self._gen_filename(key)
        if name is not None:
            outputs[key] = name
    return outputs