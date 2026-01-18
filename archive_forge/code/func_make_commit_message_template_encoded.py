import codecs
import os
import sys
from io import BytesIO, StringIO
from subprocess import call
from . import bedding, cmdline, config, osutils, trace, transport, ui
from .errors import BzrError
from .hooks import Hooks
def make_commit_message_template_encoded(working_tree, specific_files, diff=None, output_encoding='utf-8'):
    """Prepare a template file for a commit into a branch.

    Returns an encoded string.
    """
    from .diff import show_diff_trees
    template = make_commit_message_template(working_tree, specific_files)
    template = template.encode(output_encoding, 'replace')
    if diff:
        stream = BytesIO()
        show_diff_trees(working_tree.basis_tree(), working_tree, stream, specific_files, path_encoding=output_encoding)
        template = template + b'\n' + stream.getvalue()
    return template