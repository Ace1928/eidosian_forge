import re
import datetime
import numpy as np
import csv
import ctypes
def parse_data(self, data_str):
    elems = list(range(len(self.attributes)))
    escaped_string = data_str.encode().decode('unicode-escape')
    row_tuples = []
    for raw in escaped_string.split('\n'):
        row, self.dialect = split_data_line(raw, self.dialect)
        row_tuples.append(tuple([self.attributes[i].parse_data(row[i]) for i in elems]))
    return np.array(row_tuples, [(a.name, a.dtype) for a in self.attributes])