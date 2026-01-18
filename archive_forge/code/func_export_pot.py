import inspect
import os
import sys
import breezy
from . import commands as _mod_commands
from . import errors, help_topics, option
from . import plugin as _mod_plugin
from .i18n import gettext
from .trace import mutter, note
def export_pot(outf, plugin=None, include_duplicates=False):
    exporter = _PotExporter(outf, include_duplicates)
    if plugin is None:
        _standard_options(exporter)
        _command_helps(exporter)
        _error_messages(exporter)
        _help_topics(exporter)
    else:
        _command_helps(exporter, plugin)