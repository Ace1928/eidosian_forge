import codecs
import os
import sys
from io import BytesIO, StringIO
from subprocess import call
from . import bedding, cmdline, config, osutils, trace, transport, ui
from .errors import BzrError
from .hooks import Hooks
def make_commit_message_template(working_tree, specific_files):
    """Prepare a template file for a commit into a branch.

    Returns a unicode string containing the template.
    """
    from .status import show_tree_status
    status_tmp = StringIO()
    show_tree_status(working_tree, specific_files=specific_files, to_file=status_tmp, verbose=True)
    return status_tmp.getvalue()