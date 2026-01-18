import logging
import optparse
import os
import re
import shlex
import urllib.parse
from optparse import Values
from typing import (
from pip._internal.cli import cmdoptions
from pip._internal.exceptions import InstallationError, RequirementsFileParseError
from pip._internal.models.search_scope import SearchScope
from pip._internal.network.session import PipSession
from pip._internal.network.utils import raise_for_status
from pip._internal.utils.encoding import auto_decode
from pip._internal.utils.urls import get_url_scheme
def get_line_parser(finder: Optional['PackageFinder']) -> LineParser:

    def parse_line(line: str) -> Tuple[str, Values]:
        parser = build_parser()
        defaults = parser.get_default_values()
        defaults.index_url = None
        if finder:
            defaults.format_control = finder.format_control
        args_str, options_str = break_args_options(line)
        try:
            options = shlex.split(options_str)
        except ValueError as e:
            raise OptionParsingError(f'Could not split options: {options_str}') from e
        opts, _ = parser.parse_args(options, defaults)
        return (args_str, opts)
    return parse_line