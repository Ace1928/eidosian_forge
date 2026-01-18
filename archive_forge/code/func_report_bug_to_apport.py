import os
import platform
import pprint
import sys
import time
from io import StringIO
import breezy
from . import bedding, debug, osutils, plugin, trace
def report_bug_to_apport(exc_info, stderr):
    """Report a bug to apport for optional automatic filing.

    :returns: The name of the crash file, or None if we didn't write one.
    """
    import apport
    crash_filename = _write_apport_report_to_file(exc_info)
    if crash_filename is None:
        stderr.write('\napport is set to ignore crashes in this version of brz.\n')
    else:
        trace.print_exception(exc_info, stderr)
        stderr.write("\nYou can report this problem to Breezy's developers by running\n    apport-bug %s\nif a bug-reporting window does not automatically appear.\n" % crash_filename)
    return crash_filename