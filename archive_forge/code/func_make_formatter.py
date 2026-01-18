import codecs
import itertools
import re
import sys
from io import BytesIO
from typing import Callable, Dict, List
from warnings import warn
from .lazy_import import lazy_import
from breezy import (
from breezy.i18n import gettext, ngettext
from . import errors, registry
from . import revision as _mod_revision
from . import revisionspec, trace
from . import transport as _mod_transport
from .osutils import (format_date,
from .tree import InterTree, find_previous_path
def make_formatter(self, name, *args, **kwargs):
    """Construct a formatter from arguments.

        :param name: Name of the formatter to construct.  'short', 'long' and
            'line' are built-in.
        """
    return self.get(name)(*args, **kwargs)