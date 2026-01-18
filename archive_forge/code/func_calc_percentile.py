from glob import glob
from time import time_ns
import argparse
from sys import argv
from os.path import isdir
from charset_normalizer import detect
from chardet import detect as chardet_detect
from statistics import mean
from math import ceil
def calc_percentile(data, percentile):
    n = len(data)
    p = n * percentile / 100
    sorted_data = sorted(data)
    return sorted_data[int(p)] if p.is_integer() else sorted_data[int(ceil(p)) - 1]