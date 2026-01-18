import sys
import socket
import selectors
from time import monotonic as _time
import warnings
def process_rawq(self):
    """Transfer from raw queue to cooked queue.

        Set self.eof when connection is closed.  Don't block unless in
        the midst of an IAC sequence.

        """
    buf = [b'', b'']
    try:
        while self.rawq:
            c = self.rawq_getchar()
            if not self.iacseq:
                if c == theNULL:
                    continue
                if c == b'\x11':
                    continue
                if c != IAC:
                    buf[self.sb] = buf[self.sb] + c
                    continue
                else:
                    self.iacseq += c
            elif len(self.iacseq) == 1:
                if c in (DO, DONT, WILL, WONT):
                    self.iacseq += c
                    continue
                self.iacseq = b''
                if c == IAC:
                    buf[self.sb] = buf[self.sb] + c
                else:
                    if c == SB:
                        self.sb = 1
                        self.sbdataq = b''
                    elif c == SE:
                        self.sb = 0
                        self.sbdataq = self.sbdataq + buf[1]
                        buf[1] = b''
                    if self.option_callback:
                        self.option_callback(self.sock, c, NOOPT)
                    else:
                        self.msg('IAC %d not recognized' % ord(c))
            elif len(self.iacseq) == 2:
                cmd = self.iacseq[1:2]
                self.iacseq = b''
                opt = c
                if cmd in (DO, DONT):
                    self.msg('IAC %s %d', cmd == DO and 'DO' or 'DONT', ord(opt))
                    if self.option_callback:
                        self.option_callback(self.sock, cmd, opt)
                    else:
                        self.sock.sendall(IAC + WONT + opt)
                elif cmd in (WILL, WONT):
                    self.msg('IAC %s %d', cmd == WILL and 'WILL' or 'WONT', ord(opt))
                    if self.option_callback:
                        self.option_callback(self.sock, cmd, opt)
                    else:
                        self.sock.sendall(IAC + DONT + opt)
    except EOFError:
        self.iacseq = b''
        self.sb = 0
    self.cookedq = self.cookedq + buf[0]
    self.sbdataq = self.sbdataq + buf[1]