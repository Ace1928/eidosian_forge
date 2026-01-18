import sys
import os
import socket
def make_pipe():
    if sys.platform[:3] != 'win':
        p = PosixPipe()
    else:
        p = WindowsPipe()
    return p