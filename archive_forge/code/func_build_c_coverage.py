import glob
import inspect
import pickle
import re
from importlib import import_module
from os import path
from typing import IO, Any, Dict, List, Pattern, Set, Tuple
import sphinx
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.locale import __
from sphinx.util import logging
from sphinx.util.console import red  # type: ignore
from sphinx.util.inspect import safe_getattr
def build_c_coverage(self) -> None:
    c_objects = self.env.domaindata['c']['objects']
    for filename in self.c_sourcefiles:
        undoc: Set[Tuple[str, str]] = set()
        with open(filename, encoding='utf-8') as f:
            for line in f:
                for key, regex in self.c_regexes:
                    match = regex.match(line)
                    if match:
                        name = match.groups()[0]
                        if name not in c_objects:
                            for exp in self.c_ignorexps.get(key, []):
                                if exp.match(name):
                                    break
                            else:
                                undoc.add((key, name))
                        continue
        if undoc:
            self.c_undoc[filename] = undoc