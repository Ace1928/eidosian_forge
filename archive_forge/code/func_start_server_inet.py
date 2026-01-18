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
def start_server_inet(self, extra_options=()):
    """Start a brz server subprocess using the --inet option.

        :param extra_options: extra options to give the server.
        :return: a tuple with the brz process handle for passing to
            finish_brz_subprocess, a client for the server, and a transport.
        """
    args = ['serve', '--inet']
    args.extend(extra_options)
    process = self.start_brz_subprocess(args)
    url = 'bzr://localhost/'
    self.permit_url(url)
    client_medium = medium.SmartSimplePipesClientMedium(process.stdout, process.stdin, url)
    transport = remote.RemoteTransport(url, medium=client_medium)
    return (process, transport)