from __future__ import print_function
import select
import contextlib
import errno
from systemd import login
import pytest
def test_monitor():
    p = select.poll()
    with skip_oserror(errno.ENOENT):
        m = login.Monitor('machine')
        p.register(m, m.get_events())
        login.machine_names()
        p.poll(1)
        login.machine_names()