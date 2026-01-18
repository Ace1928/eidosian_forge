import errno
import os
import subprocess as sp
import sys
from ._process_common import getoutput, arg_split
from IPython.utils.encoding import DEFAULT_ENCODING
def getoutput_pexpect(self, cmd):
    """Run a command and return its stdout/stderr as a string.

        Parameters
        ----------
        cmd : str
            A command to be executed in the system shell.

        Returns
        -------
        output : str
            A string containing the combination of stdout and stderr from the
        subprocess, in whatever order the subprocess originally wrote to its
        file descriptors (so the order of the information in this string is the
        correct order as would be seen if running the command in a terminal).
        """
    import pexpect
    try:
        return pexpect.run(self.sh, args=['-c', cmd]).replace('\r\n', '\n')
    except KeyboardInterrupt:
        print('^C', file=sys.stderr, end='')