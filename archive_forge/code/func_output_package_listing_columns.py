import json
import logging
from optparse import Values
from typing import TYPE_CHECKING, Generator, List, Optional, Sequence, Tuple, cast
from pip._vendor.packaging.utils import canonicalize_name
from pip._internal.cli import cmdoptions
from pip._internal.cli.req_command import IndexGroupCommand
from pip._internal.cli.status_codes import SUCCESS
from pip._internal.exceptions import CommandError
from pip._internal.index.collector import LinkCollector
from pip._internal.index.package_finder import PackageFinder
from pip._internal.metadata import BaseDistribution, get_environment
from pip._internal.models.selection_prefs import SelectionPreferences
from pip._internal.network.session import PipSession
from pip._internal.utils.compat import stdlib_pkgs
from pip._internal.utils.misc import tabulate, write_output
def output_package_listing_columns(self, data: List[List[str]], header: List[str]) -> None:
    if len(data) > 0:
        data.insert(0, header)
    pkg_strings, sizes = tabulate(data)
    if len(data) > 0:
        pkg_strings.insert(1, ' '.join(('-' * x for x in sizes)))
    for val in pkg_strings:
        write_output(val)