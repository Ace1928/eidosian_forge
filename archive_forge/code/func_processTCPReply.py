import socket, string, types, time, select
import errno
from . import Type,Class,Opcode
from . import Lib
def processTCPReply(self):
    if self.timeout > 0:
        self.s.settimeout(self.timeout)
    else:
        self.s.settimeout(None)
    f = self.s.makefile('rb')
    try:
        header = self._readall(f, 2)
        count = Lib.unpack16bit(header)
        self.reply = self._readall(f, count)
    finally:
        f.close()
    self.time_finish = time.time()
    self.args['server'] = self.ns
    return self.processReply()