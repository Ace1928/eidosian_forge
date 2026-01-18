import sys
import socket
from socket import _GLOBAL_DEFAULT_TIMEOUT
def getmultiline(self):
    line = self.getline()
    if line[3:4] == '-':
        code = line[:3]
        while 1:
            nextline = self.getline()
            line = line + ('\n' + nextline)
            if nextline[:3] == code and nextline[3:4] != '-':
                break
    return line