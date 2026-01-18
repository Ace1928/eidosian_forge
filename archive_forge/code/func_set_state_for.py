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
def set_state_for(function_names, level):
    for fn in function_names:
        label = '.'.join(fn.split('.')[-level:])
        label_state = state.setdefault(label, {'functions': [], 'level': level})
        label_state['functions'].append(fn)