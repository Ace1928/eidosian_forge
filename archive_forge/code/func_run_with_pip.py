import datetime
import locale
import re
import subprocess
import sys
import os
from collections import namedtuple
def run_with_pip(pip):
    out = run_and_read_all(run_lambda, pip + ['list', '--format=freeze'])
    return '\n'.join((line for line in out.splitlines() if any((name in line for name in patterns))))