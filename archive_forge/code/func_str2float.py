import re
import numpy as np
from xml.dom import minidom
def str2float(string):
    numeric_const_pattern = '\n  [-+]? # optional sign\n  (?:\n    (?: \\d* \\. \\d+ ) # .1 .12 .123 etc 9.1 etc 98.1 etc\n    |\n    (?: \\d+ \\.? ) # 1. 12. 123. etc 1 12 123 etc\n  )\n  # followed by optional exponent part if desired\n  (?: [Ee] [+-]? \\d+ ) ?\n  '
    rx = re.compile(numeric_const_pattern, re.VERBOSE)
    nb = rx.findall(string)
    for i in enumerate(nb):
        nb[i[0]] = float(i[1])
    return np.array(nb)