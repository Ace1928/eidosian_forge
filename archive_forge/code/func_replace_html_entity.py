import html.entities
import re
import sys
import typing
from . import __diag__
from .core import *
from .util import (
def replace_html_entity(s, l, t):
    """Helper parser action to replace common HTML entities with their special characters"""
    return _htmlEntityMap.get(t.entity)