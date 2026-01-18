import os
import os.path
import sys
def ovs_retval_to_string(retval):
    """Many OVS functions return an int which is one of:
    - 0: no error yet
    - >0: errno value
    - EOF: end of file (not necessarily an error; depends on the function
      called)

    Returns the appropriate human-readable string."""
    if not retval:
        return ''
    if retval > 0:
        return os.strerror(retval)
    if retval == EOF:
        return 'End of file'
    return '***unknown return value: %s***' % retval