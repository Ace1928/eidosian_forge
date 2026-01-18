import re
import datetime
import numpy as np
import csv
import ctypes
def split_data_line(line, dialect=None):
    delimiters = ',\t'
    csv.field_size_limit(int(ctypes.c_ulong(-1).value // 2))
    if line[-1] == '\n':
        line = line[:-1]
    line = line.strip()
    sniff_line = line
    if not any((d in line for d in delimiters)):
        sniff_line += ','
    if dialect is None:
        dialect = csv.Sniffer().sniff(sniff_line, delimiters=delimiters)
        workaround_csv_sniffer_bug_last_field(sniff_line=sniff_line, dialect=dialect, delimiters=delimiters)
    row = next(csv.reader([line], dialect))
    return (row, dialect)