import __future__
import builtins
import ast
import collections
import contextlib
import doctest
import functools
import os
import re
import string
import sys
import warnings
from pyflakes import messages
def getNodeName(node):
    if hasattr(node, 'id'):
        return node.id
    if hasattr(node, 'name'):
        return node.name
    if hasattr(node, 'rest'):
        return node.rest