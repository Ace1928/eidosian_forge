from binascii import b2a_hex
import os
import sys
import warnings
def clear_output(wait=False):
    """Clear the output of the current cell receiving output.

    Parameters
    ----------
    wait : bool [default: false]
        Wait to clear the output until new output is available to replace it."""
    from IPython.core.interactiveshell import InteractiveShell
    if InteractiveShell.initialized():
        InteractiveShell.instance().display_pub.clear_output(wait)
    else:
        print('\x1b[2K\r', end='')
        sys.stdout.flush()
        print('\x1b[2K\r', end='')
        sys.stderr.flush()