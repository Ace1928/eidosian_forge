import os
import socket
import textwrap
import unittest
from contextlib import closing
from socket import AF_INET
from socket import AF_INET6
from socket import SOCK_DGRAM
from socket import SOCK_STREAM
import psutil
from psutil import FREEBSD
from psutil import LINUX
from psutil import MACOS
from psutil import NETBSD
from psutil import OPENBSD
from psutil import POSIX
from psutil import SUNOS
from psutil import WINDOWS
from psutil._common import supports_ipv6
from psutil._compat import PY3
from psutil.tests import AF_UNIX
from psutil.tests import HAS_CONNECTIONS_UNIX
from psutil.tests import SKIP_SYSCONS
from psutil.tests import PsutilTestCase
from psutil.tests import bind_socket
from psutil.tests import bind_unix_socket
from psutil.tests import check_connection_ntuple
from psutil.tests import create_sockets
from psutil.tests import filter_proc_connections
from psutil.tests import reap_children
from psutil.tests import retry_on_failure
from psutil.tests import serialrun
from psutil.tests import skip_on_access_denied
from psutil.tests import tcp_socketpair
from psutil.tests import unix_socketpair
from psutil.tests import wait_for_file
@retry_on_failure()
def test_multi_sockets_procs(self):
    with create_sockets() as socks:
        expected = len(socks)
    pids = []
    times = 10
    fnames = []
    for _ in range(times):
        fname = self.get_testfn()
        fnames.append(fname)
        src = textwrap.dedent('                import time, os\n                from psutil.tests import create_sockets\n                with create_sockets():\n                    with open(r\'%s\', \'w\') as f:\n                        f.write("hello")\n                    time.sleep(60)\n                ' % fname)
        sproc = self.pyrun(src)
        pids.append(sproc.pid)
    for fname in fnames:
        wait_for_file(fname)
    syscons = [x for x in psutil.net_connections(kind='all') if x.pid in pids]
    for pid in pids:
        self.assertEqual(len([x for x in syscons if x.pid == pid]), expected)
        p = psutil.Process(pid)
        self.assertEqual(len(p.connections('all')), expected)