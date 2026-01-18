import contextlib
import shutil
import sys
import tempfile
from io import BytesIO
import patiencediff
from . import (builtins, delta, diff, errors, osutils, patches, shelf,
from .i18n import gettext
def prompt_change(self, change):
    """Determine the prompt for a change to apply."""
    if change[0] == 'rename':
        vals = {'this': change[3], 'other': change[2]}
    elif change[0] == 'change kind':
        vals = {'path': change[4], 'other': change[2], 'this': change[3]}
    elif change[0] == 'modify target':
        vals = {'path': change[2], 'other': change[3], 'this': change[4]}
    else:
        vals = {'path': change[3]}
    prompt = self.vocab[change[0]] % vals
    return prompt