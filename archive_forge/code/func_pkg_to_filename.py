import sys
import re
import os
from configparser import RawConfigParser
def pkg_to_filename(pkg_name):
    return '%s.ini' % pkg_name