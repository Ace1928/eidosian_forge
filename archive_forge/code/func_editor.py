import os
import subprocess
import sys
from .error import TryNext
def editor(self, filename, linenum=None, wait=True):
    """Open the default editor at the given filename and linenumber.

    This is IPython's default editor hook, you can use it as an example to
    write your own modified one.  To set your own editor function as the
    new editor hook, call ip.set_hook('editor',yourfunc)."""
    editor = self.editor
    if linenum is None or editor == 'notepad':
        linemark = ''
    else:
        linemark = '+%d' % int(linenum)
    if ' ' in editor and os.path.isfile(editor) and (editor[0] != '"'):
        editor = '"%s"' % editor
    proc = subprocess.Popen('%s %s %s' % (editor, linemark, filename), shell=True)
    if wait and proc.wait() != 0:
        raise TryNext()