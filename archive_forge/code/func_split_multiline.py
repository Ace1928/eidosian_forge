import logging  # noqa
from collections import defaultdict
import io
import os
import re
import shlex
import sys
import traceback
import distutils.ccompiler
from distutils import errors
from distutils import log
import pkg_resources
from setuptools import dist as st_dist
from setuptools import extension
from pbr import extra_files
import pbr.hooks
def split_multiline(value):
    """Special behaviour when we have a multi line options"""
    value = [element for element in (line.strip() for line in value.split('\n')) if element and (not element.startswith('#'))]
    return value