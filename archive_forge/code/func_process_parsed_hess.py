from __future__ import annotations
import re
from collections import defaultdict
import numpy as np
def process_parsed_hess(hess_data):
    """
    Takes the information contained in a HESS file and converts it into
    the format of the machine-readable 132.0 file which can be printed
    out to be read into subsequent optimizations.
    """
    dim = int(hess_data[1].split()[1])
    hess = [[0 for _ in range(dim)] for _ in range(dim)]
    row = 0
    column = 0
    for ii, line in enumerate(hess_data):
        if ii not in [0, 1, len(hess_data) - 1]:
            split_line = line.split()
            for val in split_line:
                num = float(val)
                hess[row][column] = num
                if row == column:
                    row += 1
                    column = 0
                else:
                    hess[column][row] = num
                    column += 1
    processed_hess_data = []
    for ii in range(dim):
        for jj in range(dim):
            processed_hess_data.append(hess[ii][jj])
    return processed_hess_data