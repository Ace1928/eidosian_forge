import os
import platform
import pprint
import sys
import time
from io import StringIO
import breezy
from . import bedding, debug, osutils, plugin, trace
def report_bug_legacy(exc_info, err_file):
    """Report a bug by just printing a message to the user."""
    trace.print_exception(exc_info, err_file)
    err_file.write('\n')
    import textwrap

    def print_wrapped(l):
        err_file.write(textwrap.fill(l, width=78, subsequent_indent='    ') + '\n')
    print_wrapped('brz %s on python %s (%s)\n' % (breezy.__version__, breezy._format_version_tuple(sys.version_info), platform.platform(aliased=1)))
    print_wrapped('arguments: %r\n' % sys.argv)
    print_wrapped(textwrap.fill('plugins: ' + plugin.format_concise_plugin_list(), width=78, subsequent_indent='    ') + '\n')
    print_wrapped('encoding: {!r}, fsenc: {!r}, lang: {!r}\n'.format(osutils.get_user_encoding(), sys.getfilesystemencoding(), os.environ.get('LANG')))
    err_file.write('\n*** Breezy has encountered an internal error.  This probably indicates a\n    bug in Breezy.  You can help us fix it by filing a bug report at\n        https://bugs.launchpad.net/brz/+filebug\n    including this traceback and a description of the problem.\n')