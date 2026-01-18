import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast
import srsly
from catalogue import RegistryError
from thinc.api import Config
from wasabi import MarkdownRenderer, Printer, get_raw_input
from .. import about, util
from ..compat import importlib_metadata
from ..schemas import ModelMetaSchema, validate
from ._util import SDIST_SUFFIX, WHEEL_SUFFIX, Arg, Opt, app, string_to_list
import io
import json
from os import path, walk
from shutil import copy
from setuptools import setup
from pathlib import Path
from spacy.util import load_model_from_init_py, get_model_meta
def get_third_party_dependencies(config: Config, exclude: List[str]=util.SimpleFrozenList()) -> List[str]:
    """If the config includes references to registered functions that are
    provided by third-party packages (spacy-transformers, other libraries), we
    want to include them in meta["requirements"] so that the package specifies
    them as dependencies and the user won't have to do it manually.

    We do this by:
    - traversing the config to check for registered function (@ keys)
    - looking up the functions and getting their module
    - looking up the module version and generating an appropriate version range

    config (Config): The pipeline config.
    exclude (list): List of packages to exclude (e.g. that already exist in meta).
    RETURNS (list): The versioned requirements.
    """
    own_packages = ('spacy', 'spacy-legacy', 'spacy-nightly', 'thinc', 'srsly')
    distributions = util.packages_distributions()
    funcs = defaultdict(set)
    for section in ('nlp', 'components'):
        for path, value in util.walk_dict(config[section]):
            if path[-1].startswith('@'):
                funcs[path[-1][1:]].add(value)
    for component in config.get('components', {}).values():
        if 'factory' in component:
            funcs['factories'].add(component['factory'])
    modules = set()
    lang = config['nlp']['lang']
    for reg_name, func_names in funcs.items():
        for func_name in func_names:
            try:
                func_info = util.registry.find(reg_name, lang + '.' + func_name)
            except RegistryError:
                try:
                    func_info = util.registry.find(reg_name, func_name)
                except RegistryError as regerr:
                    raise regerr from None
            module_name = func_info.get('module')
            if module_name:
                modules.add(func_info['module'].split('.')[0])
    dependencies = []
    for module_name in modules:
        if module_name in distributions:
            dist = distributions.get(module_name)
            if dist:
                pkg = dist[0]
                if pkg in own_packages or pkg in exclude:
                    continue
                version = util.get_package_version(pkg)
                version_range = util.get_minor_version_range(version)
                dependencies.append(f'{pkg}{version_range}')
    return dependencies