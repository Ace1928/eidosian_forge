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
def show_tags(options: Values) -> None:
    tag_limit = 10
    target_python = make_target_python(options)
    tags = target_python.get_sorted_tags()
    formatted_target = target_python.format_given()
    suffix = ''
    if formatted_target:
        suffix = f' (target: {formatted_target})'
    msg = f'Compatible tags: {len(tags)}{suffix}'
    logger.info(msg)
    if options.verbose < 1 and len(tags) > tag_limit:
        tags_limited = True
        tags = tags[:tag_limit]
    else:
        tags_limited = False
    with indent_log():
        for tag in tags:
            logger.info(str(tag))
        if tags_limited:
            msg = f'...\n[First {tag_limit} tags shown. Pass --verbose to show all.]'
            logger.info(msg)