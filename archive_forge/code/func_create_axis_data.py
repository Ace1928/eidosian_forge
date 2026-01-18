from __future__ import (absolute_import, division, print_function)
import os
import argparse
import csv
from collections import namedtuple
def create_axis_data(filename, relative=False):
    x_base = None if relative else 0
    axis_name, dummy = os.path.splitext(os.path.basename(filename))
    dates = []
    names = []
    values = []
    with open(filename) as f:
        reader = csv.reader(f)
        for row in reader:
            if x_base is None:
                x_base = float(row[0])
            dates.append(mdates.epoch2num(float(row[0]) - x_base))
            names.append(row[1])
            values.append(float(row[3]))
    return Data(axis_name, dates, names, values)