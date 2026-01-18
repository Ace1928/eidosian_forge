import sys
from collections.abc import MutableSequence
import re
from textwrap import dedent
from keyword import iskeyword
import flask
from ._grouping import grouping_len, map_grouping
from .development.base_component import Component
from . import exceptions
from ._utils import (
def check_for_duplicate_pathnames(registry):
    path_to_module = {}
    for page in registry.values():
        if page['path'] not in path_to_module:
            path_to_module[page['path']] = [page['module']]
        else:
            path_to_module[page['path']].append(page['module'])
    for modules in path_to_module.values():
        if len(modules) > 1:
            raise Exception(f'modules {modules} have duplicate paths')