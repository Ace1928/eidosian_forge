import sys
from argparse import ArgumentParser, FileType
from textwrap import indent
from logging import DEBUG, INFO, WARN, ERROR
from typing import Optional
import warnings
from lark import Lark, logger
def make_warnings_comments():
    warnings.showwarning = showwarning_as_comment