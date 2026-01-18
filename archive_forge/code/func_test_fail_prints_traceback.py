import datetime
import io
import os
import re
import signal
import sys
import threading
from unittest import mock
import fixtures
import greenlet
from oslotest import base
import oslo_config
from oslo_config import fixture
from oslo_reports import guru_meditation_report as gmr
from oslo_reports.models import with_default_views as mwdv
from oslo_reports import opts
@mock.patch.object(gmr.TextGuruMeditation, 'run')
def test_fail_prints_traceback(self, run_mock):

    class RunFail(Exception):
        pass
    run_mock.side_effect = RunFail()
    gmr.TextGuruMeditation.setup_autorun(FakeVersionObj())
    self.old_stderr = sys.stderr
    sys.stderr = io.StringIO()
    os.kill(os.getpid(), signal.SIGUSR2)
    self.assertIn('RunFail', sys.stderr.getvalue())