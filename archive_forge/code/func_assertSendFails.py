from io import BytesIO
from ... import branch, merge_directive, tests
from ...bzr.bundle import serializer
from ...controldir import ControlDir
from ...transport import memory
from .. import scenarios
def assertSendFails(self, args):
    out, err = self.run_send(args, rc=3, err_re=self._default_errors)
    self.assertContainsRe(err, self._default_additional_error)