import sys
import socket
from socket import _GLOBAL_DEFAULT_TIMEOUT
def parse257(resp):
    """Parse the '257' response for a MKD or PWD request.
    This is a response to a MKD or PWD request: a directory name.
    Returns the directoryname in the 257 reply."""
    if resp[:3] != '257':
        raise error_reply(resp)
    if resp[3:5] != ' "':
        return ''
    dirname = ''
    i = 5
    n = len(resp)
    while i < n:
        c = resp[i]
        i = i + 1
        if c == '"':
            if i >= n or resp[i] != '"':
                break
            i = i + 1
        dirname = dirname + c
    return dirname