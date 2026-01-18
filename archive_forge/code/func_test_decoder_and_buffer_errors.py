import os
import time
import sys
from io import StringIO, BytesIO
from pyomo.common.log import LoggingIntercept
import pyomo.common.unittest as unittest
from pyomo.common.tempfiles import TempfileManager
import pyomo.common.tee as tee
def test_decoder_and_buffer_errors(self):
    ref = 'Hello, ©'
    bytes_ref = ref.encode()
    log = StringIO()
    with LoggingIntercept(log):
        with tee.TeeStream(encoding='utf-8') as t:
            os.write(t.STDOUT.fileno(), bytes_ref[:-1])
    self.assertEqual(log.getvalue(), "Stream handle closed with a partial line in the output buffer that was not emitted to the output stream(s):\n\t'Hello, '\nStream handle closed with un-decoded characters in the decoder buffer that was not emitted to the output stream(s):\n\tb'\\xc2'\n")
    out = StringIO()
    log = StringIO()
    with LoggingIntercept(log):
        with tee.TeeStream(out) as t:
            out.close()
            t.STDOUT.write('hi\n')
    self.assertRegex(log.getvalue(), "^Output stream \\(<.*?>\\) closed before all output was written to it. The following was left in the output buffer:\\n\\t'hi\\\\n'\\n$")