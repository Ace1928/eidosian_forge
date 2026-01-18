import os
import platform
import subprocess
import errno
import time
import sys
import unittest
import tempfile
def read_async(self, wait=0.1, e=1, tr=5, stderr=0):
    if tr < 1:
        tr = 1
    x = time.time() + wait
    y = []
    r = ''
    pr = self.recv
    if stderr:
        pr = self.recv_err
    while time.time() < x or r:
        r = pr()
        if r is None:
            if e:
                raise Exception('Other end disconnected!')
            else:
                break
        elif r:
            y.append(r)
        else:
            time.sleep(max((x - time.time()) / tr, 0))
    return ''.join(y)