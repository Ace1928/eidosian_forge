import importlib.resources
import locale
import logging
import os
import sys
from optparse import Values
from types import ModuleType
from typing import Any, Dict, List, Optional
import pip._vendor
from pip._vendor.certifi import where
from pip._vendor.packaging.version import parse as parse_version
from pip._internal.cli import cmdoptions
from pip._internal.cli.base_command import Command
from pip._internal.cli.cmdoptions import make_target_python
from pip._internal.cli.status_codes import SUCCESS
from pip._internal.configuration import Configuration
from pip._internal.metadata import get_environment
from pip._internal.utils.logging import indent_log
from pip._internal.utils.misc import get_pip_version
def create_vendor_txt_map() -> Dict[str, str]:
    with importlib.resources.open_text('pip._vendor', 'vendor.txt') as f:
        lines = [line.strip().split(' ', 1)[0] for line in f.readlines() if '==' in line]
    return dict((line.split('==', 1) for line in lines))