import sys
import os.path
import csv
from docutils import io, nodes, statemachine, utils
from docutils.utils.error_reporting import SafeString
from docutils.utils import SystemMessagePropagation
from docutils.parsers.rst import Directive
from docutils.parsers.rst import directives
def encode_for_csv(s):
    return s