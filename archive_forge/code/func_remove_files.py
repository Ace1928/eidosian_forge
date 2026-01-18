import errno
import os
import re
from .lazy_import import lazy_import
from breezy import (
from breezy.i18n import gettext, ngettext
from . import commands, errors, option, osutils, registry, trace
def remove_files(self, tree):
    """Remove the THIS, BASE and OTHER files for listed conflicts"""
    for conflict in self:
        if not conflict.has_files:
            continue
        conflict.cleanup(tree)