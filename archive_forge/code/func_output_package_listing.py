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
def output_package_listing(self, packages: '_ProcessedDists', options: Values) -> None:
    packages = sorted(packages, key=lambda dist: dist.canonical_name)
    if options.list_format == 'columns' and packages:
        data, header = format_for_columns(packages, options)
        self.output_package_listing_columns(data, header)
    elif options.list_format == 'freeze':
        for dist in packages:
            if options.verbose >= 1:
                write_output('%s==%s (%s)', dist.raw_name, dist.version, dist.location)
            else:
                write_output('%s==%s', dist.raw_name, dist.version)
    elif options.list_format == 'json':
        write_output(format_for_json(packages, options))