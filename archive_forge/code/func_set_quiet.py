import os
import ovs.util
import ovs.vlog
def set_quiet(self, quiet):
    """If 'quiet' is true, this object will log informational messages at
        debug level, by default keeping them out of log files.  This is
        appropriate if the connection is one that is expected to be
        short-lived, so that the log messages are merely distracting.

        If 'quiet' is false, this object logs informational messages at info
        level.  This is the default.

        This setting has no effect on the log level of debugging, warning, or
        error messages."""
    if quiet:
        self.info_level = vlog.dbg
    else:
        self.info_level = vlog.info