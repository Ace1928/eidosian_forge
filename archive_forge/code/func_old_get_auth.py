import fcntl
import os
import platform
import re
import socket
from Xlib import error, xauth
def old_get_auth(sock, dname, host, dno):
    auth_name = auth_data = ''
    try:
        data = os.popen('xauth list %s 2>/dev/null' % dname).read()
        lines = data.split('\n')
        if len(lines) >= 1:
            parts = lines[0].split(None, 2)
            if len(parts) == 3:
                auth_name = parts[1]
                hexauth = parts[2]
                auth = ''
                for i in range(0, len(hexauth), 2):
                    auth = auth + chr(int(hexauth[i:i + 2], 16))
                auth_data = auth
    except os.error:
        pass
    return (auth_name, auth_data)