from __future__ import (absolute_import, division, print_function)
import os
import argparse
import csv
from collections import namedtuple
def task_start_ticks(dates, names):
    item = None
    ret = []
    for i, name in enumerate(names):
        if name == item:
            continue
        item = name
        ret.append((dates[i], name))
    return ret