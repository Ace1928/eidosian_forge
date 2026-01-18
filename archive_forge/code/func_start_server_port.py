import signal
import sys
import threading
from _thread import interrupt_main  # type: ignore
from ... import builtins, config, errors, osutils
from ... import revision as _mod_revision
from ... import trace, transport, urlutils
from ...branch import Branch
from ...bzr.smart import client, medium
from ...bzr.smart.server import BzrServerFactory, SmartTCPServer
from ...controldir import ControlDir
from ...transport import remote
from .. import TestCaseWithMemoryTransport, TestCaseWithTransport
def start_server_port(self, extra_options=()):
    """Start a brz server subprocess.

        :param extra_options: extra options to give the server.
        :return: a tuple with the brz process handle for passing to
            finish_brz_subprocess, and the base url for the server.
        """
    args = ['serve', '--listen', 'localhost', '--port', '0']
    args.extend(extra_options)
    process = self.start_brz_subprocess(args, skip_if_plan_to_signal=True)
    port_line = process.stderr.readline()
    prefix = b'listening on port: '
    self.assertStartsWith(port_line, prefix)
    port = int(port_line[len(prefix):])
    url = 'bzr://localhost:%d/' % port
    self.permit_url(url)
    return (process, url)