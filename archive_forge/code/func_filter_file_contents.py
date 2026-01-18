import enum
import glob
import logging
import math
import os
import operator
import re
import subprocess
import sys
from io import StringIO
from unittest import *
import unittest as _unittest
import pytest as pytest
from pyomo.common.collections import Mapping, Sequence
from pyomo.common.dependencies import attempt_import, check_min_version
from pyomo.common.errors import InvalidValueError
from pyomo.common.fileutils import import_file
from pyomo.common.log import LoggingIntercept, pyomo_formatter
from pyomo.common.tee import capture_output
from unittest import mock
def filter_file_contents(self, lines, abstol=None):
    filtered = []
    deprecated = None
    for line in lines:
        if line.startswith('WARNING: DEPRECATED:'):
            deprecated = ''
        if deprecated is not None:
            deprecated += line
            if re.search('\\(called\\s+from[^)]+\\)', deprecated):
                deprecated = None
            continue
        if not line or self.filter_fcn(line):
            continue
        if 'seconds' in line:
            s = line.find('seconds') + 7
            line = line[s:]
        item_list = []
        items = line.strip().split()
        for i in items:
            if '.inf' in i:
                i = i.replace('.inf', 'inf')
            if 'null' in i:
                i = i.replace('null', 'None')
            try:
                item_list.append(float(i))
            except:
                item_list.append(i)
        if len(item_list) == 2 and item_list[0] == 'Value:' and (type(item_list[1]) is float) and (abs(item_list[1]) < (abstol or 0)) and (len(filtered[-1]) == 1) and (filtered[-1][0][-1] == ':'):
            filtered.pop()
        else:
            filtered.append(item_list)
    return filtered