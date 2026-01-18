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
def split_csv(value):
    """Special behaviour when we have a comma separated options"""
    value = [element for element in (chunk.strip() for chunk in value.split(',')) if element]
    return value