import os
import os.path
import sys
def ovs_error(err_no, message, vlog=None):
    """Prints 'message' on stderr and emits an ERROR level log message to
    'vlog' if supplied.  If 'err_no' is nonzero, then it is formatted with
    ovs_retval_to_string() and appended to the message inside parentheses.

    'message' should not end with a new-line, because this function will add
    one itself."""
    err_msg = '%s: %s' % (PROGRAM_NAME, message)
    if err_no:
        err_msg += ' (%s)' % ovs_retval_to_string(err_no)
    sys.stderr.write('%s\n' % err_msg)
    if vlog:
        vlog.err(err_msg)