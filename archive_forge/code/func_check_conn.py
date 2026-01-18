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
def check_conn(proc, conn, family, type, laddr, raddr, status, kinds):
    all_kinds = ('all', 'inet', 'inet4', 'inet6', 'tcp', 'tcp4', 'tcp6', 'udp', 'udp4', 'udp6')
    check_connection_ntuple(conn)
    self.assertEqual(conn.family, family)
    self.assertEqual(conn.type, type)
    self.assertEqual(conn.laddr, laddr)
    self.assertEqual(conn.raddr, raddr)
    self.assertEqual(conn.status, status)
    for kind in all_kinds:
        cons = proc.connections(kind=kind)
        if kind in kinds:
            self.assertNotEqual(cons, [])
        else:
            self.assertEqual(cons, [])
    if HAS_CONNECTIONS_UNIX:
        self.compare_procsys_connections(proc.pid, [conn])