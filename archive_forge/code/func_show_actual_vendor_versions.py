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
def show_actual_vendor_versions(vendor_txt_versions: Dict[str, str]) -> None:
    """Log the actual version and print extra info if there is
    a conflict or if the actual version could not be imported.
    """
    for module_name, expected_version in vendor_txt_versions.items():
        extra_message = ''
        actual_version = get_vendor_version_from_module(module_name)
        if not actual_version:
            extra_message = ' (Unable to locate actual module version, using vendor.txt specified version)'
            actual_version = expected_version
        elif parse_version(actual_version) != parse_version(expected_version):
            extra_message = f' (CONFLICT: vendor.txt suggests version should be {expected_version})'
        logger.info('%s==%s%s', module_name, actual_version, extra_message)