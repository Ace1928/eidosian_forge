import argparse
import io
import logging
import os
import platform
import re
import shutil
import sys
import subprocess
from . import envvar
from .deprecation import deprecated
from .errors import DeveloperError
import pyomo.common
from pyomo.common.dependencies import attempt_import
def get_binary_file(self, url):
    """Retrieve the specified url and write as a binary file"""
    return self.get_file(url, binary=True)