import os
import shlex
import subprocess
import sys
from IPython import get_ipython
from IPython.core.error import TryNext
from IPython.utils import py3compat
def install_editor(template, wait=False):
    """Installs the editor that is called by IPython for the %edit magic.

    This overrides the default editor, which is generally set by your EDITOR
    environment variable or is notepad (windows) or vi (linux). By supplying a
    template string `run_template`, you can control how the editor is invoked
    by IPython -- (e.g. the format in which it accepts command line options)

    Parameters
    ----------
    template : basestring
        run_template acts as a template for how your editor is invoked by
        the shell. It should contain '{filename}', which will be replaced on
        invocation with the file name, and '{line}', $line by line number
        (or 0) to invoke the file with.
    wait : bool
        If `wait` is true, wait until the user presses enter before returning,
        to facilitate non-blocking editors that exit immediately after
        the call.
    """

    def call_editor(self, filename, line=0):
        if line is None:
            line = 0
        cmd = template.format(filename=shlex.quote(filename), line=line)
        print('>', cmd)
        if sys.platform.startswith('win'):
            cmd = shlex.split(cmd)
        proc = subprocess.Popen(cmd, shell=True)
        if proc.wait() != 0:
            raise TryNext()
        if wait:
            py3compat.input('Press Enter when done editing:')
    get_ipython().set_hook('editor', call_editor)
    get_ipython().editor = template