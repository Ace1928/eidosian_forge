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
def rm_action():
    """TODO: merge with clean_action (@pgervais)"""
    parser = ArgumentParser(usage='mprof rm [options] numbers_or_filenames')
    parser.add_argument('--version', action='version', version=mp.__version__)
    parser.add_argument('--dry-run', dest='dry_run', default=False, action='store_true', help='Show what will be done, without actually doing it.')
    parser.add_argument('numbers_or_filenames', nargs='*', help='numbers or filenames removed')
    args = parser.parse_args()
    if len(args.numbers_or_filenames) == 0:
        print('A profile to remove must be provided (number or filename)')
        sys.exit(1)
    filenames = get_profile_filenames(args.numbers_or_filenames)
    if args.dry_run:
        print('Files to be removed: ')
        for filename in filenames:
            print(filename)
    else:
        for filename in filenames:
            os.remove(filename)