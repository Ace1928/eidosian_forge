import asyncio
import enum
import json
import pathlib
import re
import shutil
import subprocess
import sys
from functools import lru_cache
from typing import (
from traitlets import Any as Any_
from traitlets import Instance
from traitlets import List as List_
from traitlets import Unicode, default
from traitlets.config import LoggingConfigurable
def find_node_module(self, *path_frag):
    """look through the node_module roots to find the given node module"""
    all_roots = self.extra_node_roots + self.node_roots
    found = None
    for candidate_root in all_roots:
        candidate = pathlib.Path(candidate_root, 'node_modules', *path_frag)
        self.log.debug('Checking for %s', candidate)
        if candidate.exists():
            found = str(candidate)
            break
    if found is None:
        self.log.debug('{} not found in node_modules of {}'.format(pathlib.Path(*path_frag), all_roots))
    return found