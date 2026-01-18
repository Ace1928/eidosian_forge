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
def format_for_json(packages: '_ProcessedDists', options: Values) -> str:
    data = []
    for dist in packages:
        info = {'name': dist.raw_name, 'version': str(dist.version)}
        if options.verbose >= 1:
            info['location'] = dist.location or ''
            info['installer'] = dist.installer
        if options.outdated:
            info['latest_version'] = str(dist.latest_version)
            info['latest_filetype'] = dist.latest_filetype
        editable_project_location = dist.editable_project_location
        if editable_project_location:
            info['editable_project_location'] = editable_project_location
        data.append(info)
    return json.dumps(data)