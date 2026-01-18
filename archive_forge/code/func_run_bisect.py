import sys
from . import revision as _mod_revision
from .commands import Command
from .controldir import ControlDir
from .errors import CommandError
from .option import Option
from .trace import note
def run_bisect(self, controldir, script):
    import subprocess
    note('Starting bisect.')
    self.start(controldir)
    while True:
        try:
            process = subprocess.Popen(script, shell=True)
            process.wait()
            retcode = process.returncode
            if retcode == 0:
                done = self._set_state(controldir, None, 'yes')
            elif retcode == 125:
                break
            else:
                done = self._set_state(controldir, None, 'no')
            if done:
                break
        except RuntimeError:
            break