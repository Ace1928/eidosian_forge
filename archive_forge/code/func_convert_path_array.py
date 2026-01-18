import math
import warnings
import matplotlib.dates
def convert_path_array(path_array):
    symbols = list()
    for path in path_array:
        symbols += [convert_path(path)]
    if len(symbols) == 1:
        return symbols[0]
    else:
        return symbols