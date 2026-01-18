import glob
import os
import os.path as osp
import sys
import re
import copy
import time
import math
import logging
import itertools
from ast import literal_eval
from collections import defaultdict
from argparse import ArgumentParser, ArgumentError, REMAINDER, RawTextHelpFormatter
import importlib
import memory_profiler as mp
def add_timestamp_rectangle(ax, x0, x1, y0, y1, func_name, color='none'):
    rect = ax.fill_betweenx((y0, y1), x0, x1, color=color, alpha=0.5, linewidth=1)
    text = ax.text(x0, y1, func_name, horizontalalignment='left', verticalalignment='top', color=(0, 0, 0, 0))
    return (rect, text)