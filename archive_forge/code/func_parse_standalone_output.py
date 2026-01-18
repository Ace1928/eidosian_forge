from __future__ import (absolute_import, division, print_function)
import sys
import numpy as np
def parse_standalone_output(lines):
    outs = []
    tout, yout, params = (None, None, None)
    for line in lines:
        if line.startswith('{'):
            outs.append((tout, yout, params, eval(line)))
            tout, yout = (None, None)
        elif tout is None:
            tout, yout = ([], [])
            params = line.split()
        else:
            items = line.split()
            tout.append(items[0])
            yout.append(items[1:])
    return [(np.array(_t), np.array(_y), np.asarray(_p), _nfo) for _t, _y, _p, _nfo in outs]