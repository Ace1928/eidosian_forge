import shutil
from sympy.external import import_module
from sympy.testing.pytest import skip
from sympy.utilities._compilation.compilation import compile_link_import_strings
import numpy as np
def npy(data, lim=350.0):
    return data / ((data / lim) ** 8 + 1) ** (1 / 8.0)