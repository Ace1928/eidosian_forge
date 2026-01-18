from copy import deepcopy
from abc import ABC, abstractmethod
from types import ModuleType
from typing import (
import sys
import token, tokenize
import os
from os import path
from collections import defaultdict
from functools import partial
from argparse import ArgumentParser
import lark
from lark.tools import lalr_argparser, build_lalr, make_warnings_comments
from lark.grammar import Rule
from lark.lexer import TerminalDef
def output_decompress(name):
    output('%(name)s = pickle.loads(zlib.decompress(base64.b64decode(%(name)s)))' % locals())