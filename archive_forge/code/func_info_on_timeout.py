from __future__ import annotations
import os
import subprocess
import sys
import threading
import time
import debugpy
from debugpy import adapter
from debugpy.common import json, log, messaging, sockets
from debugpy.adapter import components, sessions
import traceback
import io
def info_on_timeout():
    nonlocal output_collected
    taking_longer_than_expected = False
    initial_time = time.time()
    while True:
        time.sleep(1)
        returncode = injector.poll()
        if returncode is not None:
            if returncode != 0:
                on_output('stderr', 'Attach to PID failed.\n\n')
                old = output_collected
                output_collected = []
                contents = ''.join(old)
                on_output('stderr', ''.join(contents))
            break
        elapsed = time.time() - initial_time
        on_output('stdout', 'Attaching to PID: %s (elapsed: %.2fs).\n' % (pid, elapsed))
        if not taking_longer_than_expected:
            if elapsed > 10:
                taking_longer_than_expected = True
                if sys.platform in ('linux', 'linux2'):
                    on_output('stdout', '\nThe attach to PID is taking longer than expected.\n')
                    on_output('stdout', "On Linux it's possible to customize the value of\n")
                    on_output('stdout', '`PYDEVD_GDB_SCAN_SHARED_LIBRARIES` so that fewer libraries.\n')
                    on_output('stdout', 'are scanned when searching for the needed symbols.\n\n')
                    on_output('stdout', 'i.e.: set in your environment variables (and restart your editor/client\n')
                    on_output('stdout', 'so that it picks up the updated environment variable value):\n\n')
                    on_output('stdout', 'PYDEVD_GDB_SCAN_SHARED_LIBRARIES=libdl, libltdl, libc, libfreebl3\n\n')
                    on_output('stdout', '-- the actual library may be different (the gdb output typically\n')
                    on_output('stdout', '-- writes the libraries that will be used, so, it should be possible\n')
                    on_output('stdout', "-- to test other libraries if the above doesn't work).\n\n")
        if taking_longer_than_expected:
            old = output_collected
            output_collected = []
            contents = ''.join(old)
            if contents:
                on_output('stderr', contents)