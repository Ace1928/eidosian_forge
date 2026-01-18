import os, sys, time, re, calendar
import py
import subprocess
from py._path import common
def fixlocale():
    if sys.platform != 'win32':
        return 'LC_ALL=C '
    return ''