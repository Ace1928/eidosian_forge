from io import BytesIO
from ... import branch, merge_directive, tests
from ...bzr.bundle import serializer
from ...controldir import ControlDir
from ...transport import memory
from .. import scenarios
def run_send(self, args, cmd=None, rc=0, wd=None, err_re=None):
    if cmd is None:
        cmd = self._default_command
    if wd is None:
        wd = self._default_wd
    if err_re is None:
        err_re = []
    return self.run_bzr(cmd + args, retcode=rc, working_dir=wd, error_regexes=err_re)