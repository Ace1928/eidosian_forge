import re
import numpy as np
@property
def python_format(self):
    return '%' + str(self.width - 1) + '.' + str(self.significand) + 'E'