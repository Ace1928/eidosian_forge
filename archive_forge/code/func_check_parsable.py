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
def check_parsable(self, path, contents, lineno=1):
    """Check the given contents to verify they can be parsed as YAML."""
    try:
        yaml.load(contents, Loader=TestLoader)
    except MarkedYAMLError as ex:
        self.messages += [{'code': 'unparsable-with-libyaml', 'message': '%s - %s' % (ex.args[0], ex.args[2]), 'path': path, 'line': ex.problem_mark.line + lineno, 'column': ex.problem_mark.column + 1, 'level': 'error'}]