import socket, string, types, time, select
import errno
from . import Type,Class,Opcode
from . import Lib
def processReply(self):
    self.args['elapsed'] = (self.time_finish - self.time_start) * 1000
    if not self.resulttype:
        u = Lib.Munpacker(self.reply)
    elif self.resulttype == 'default':
        u = Lib.MunpackerDefault(self.reply)
    elif self.resulttype == 'binary':
        u = Lib.MunpackerBinary(self.reply)
    elif self.resulttype == 'text':
        u = Lib.MunpackerText(self.reply)
    elif self.resulttype == 'integer':
        u = Lib.MunpackerInteger(self.reply)
    else:
        raise SyntaxError('Unknown resulttype: ' + self.resulttype)
    r = Lib.DnsResult(u, self.args)
    r.args = self.args
    return r