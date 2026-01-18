import contextlib
import shutil
import sys
import tempfile
from io import BytesIO
import patiencediff
from . import (builtins, delta, diff, errors, osutils, patches, shelf,
from .i18n import gettext
def shelved_id(self, shelf_id):
    """Report the id changes were shelved to."""
    trace.note(gettext('Changes shelved with id "%d".') % shelf_id)