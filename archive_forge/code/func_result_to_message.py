from __future__ import annotations
import ast
import json
import os
import re
import sys
import typing as t
import yaml
from yaml.resolver import Resolver
from yaml.constructor import SafeConstructor
from yaml.error import MarkedYAMLError
from yaml.cyaml import CParser
from yamllint import linter
from yamllint.config import YamlLintConfig
@staticmethod
def result_to_message(result, path, line_offset=0, prefix=''):
    """Convert the given result to a dictionary and return it."""
    if prefix:
        prefix = '%s: ' % prefix
    return dict(code=result.rule or result.level, message=prefix + result.desc, path=path, line=result.line + line_offset, column=result.column, level=result.level)