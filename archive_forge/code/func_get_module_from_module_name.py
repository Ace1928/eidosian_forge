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
def get_module_from_module_name(module_name: str) -> Optional[ModuleType]:
    module_name = module_name.lower().replace('-', '_')
    if module_name == 'setuptools':
        module_name = 'pkg_resources'
    try:
        __import__(f'pip._vendor.{module_name}', globals(), locals(), level=0)
        return getattr(pip._vendor, module_name)
    except ImportError:
        if module_name == 'truststore' and sys.version_info < (3, 10):
            return None
        raise