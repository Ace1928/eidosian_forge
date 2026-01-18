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
def build_py_coverage(self) -> None:
    objects = self.env.domaindata['py']['objects']
    modules = self.env.domaindata['py']['modules']
    skip_undoc = self.config.coverage_skip_undoc_in_source
    for mod_name in modules:
        ignore = False
        for exp in self.mod_ignorexps:
            if exp.match(mod_name):
                ignore = True
                break
        if ignore or self.ignore_pyobj(mod_name):
            continue
        try:
            mod = import_module(mod_name)
        except ImportError as err:
            logger.warning(__('module %s could not be imported: %s'), mod_name, err)
            self.py_undoc[mod_name] = {'error': err}
            continue
        funcs = []
        classes: Dict[str, List[str]] = {}
        for name, obj in inspect.getmembers(mod):
            if name[0] == '_':
                continue
            if not hasattr(obj, '__module__'):
                continue
            if obj.__module__ != mod_name:
                continue
            full_name = '%s.%s' % (mod_name, name)
            if self.ignore_pyobj(full_name):
                continue
            if inspect.isfunction(obj):
                if full_name not in objects:
                    for exp in self.fun_ignorexps:
                        if exp.match(name):
                            break
                    else:
                        if skip_undoc and (not obj.__doc__):
                            continue
                        funcs.append(name)
            elif inspect.isclass(obj):
                for exp in self.cls_ignorexps:
                    if exp.match(name):
                        break
                else:
                    if full_name not in objects:
                        if skip_undoc and (not obj.__doc__):
                            continue
                        classes[name] = []
                        continue
                    attrs: List[str] = []
                    for attr_name in dir(obj):
                        if attr_name not in obj.__dict__:
                            continue
                        try:
                            attr = safe_getattr(obj, attr_name)
                        except AttributeError:
                            continue
                        if not (inspect.ismethod(attr) or inspect.isfunction(attr)):
                            continue
                        if attr_name[0] == '_':
                            continue
                        if skip_undoc and (not attr.__doc__):
                            continue
                        full_attr_name = '%s.%s' % (full_name, attr_name)
                        if self.ignore_pyobj(full_attr_name):
                            continue
                        if full_attr_name not in objects:
                            attrs.append(attr_name)
                    if attrs:
                        classes[name] = attrs
        self.py_undoc[mod_name] = {'funcs': funcs, 'classes': classes}