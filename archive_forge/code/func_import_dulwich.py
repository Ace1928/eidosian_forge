import os
import sys
from .. import __version__ as breezy_version  # noqa: F401
from .. import errors as brz_errors
from .. import trace, urlutils, version_info
from ..commands import plugin_cmds
from ..controldir import ControlDirFormat, Prober, format_registry
from ..controldir import \
from ..transport import (register_lazy_transport, register_transport_proto,
from ..revisionspec import RevisionSpec_dwim, revspec_registry
from ..hooks import install_lazy_named_hook
from ..location import hooks as location_hooks
from ..repository import format_registry as repository_format_registry
from ..repository import \
from ..branch import network_format_registry as branch_network_format_registry
from ..branch import format_registry as branch_format_registry
from ..workingtree import format_registry as workingtree_format_registry
from ..diff import format_registry as diff_format_registry
from ..send import format_registry as send_format_registry
from ..directory_service import directories
from ..help_topics import topic_registry
from ..foreign import foreign_vcs_registry
from ..config import Option, bool_from_store, option_registry
def import_dulwich():
    try:
        from dulwich import __version__ as dulwich_version
    except ModuleNotFoundError:
        raise brz_errors.DependencyNotPresent('dulwich', 'bzr-git: Please install dulwich, https://www.dulwich.io/')
    else:
        if dulwich_version < dulwich_minimum_version:
            raise brz_errors.DependencyNotPresent('dulwich', 'bzr-git: Dulwich is too old; at least %d.%d.%d is required' % dulwich_minimum_version)