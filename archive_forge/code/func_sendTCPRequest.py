import socket, string, types, time, select
import errno
from . import Type,Class,Opcode
from . import Lib
def sendTCPRequest(self, server):
    """ do the work of sending a TCP request """
    first_socket_error = None
    self.response = None
    for self.ns in server:
        try:
            if self.ns.count(':'):
                if hasattr(socket, 'has_ipv6') and socket.has_ipv6:
                    self.socketInit(socket.AF_INET6, socket.SOCK_STREAM)
                else:
                    continue
            else:
                self.socketInit(socket.AF_INET, socket.SOCK_STREAM)
            try:
                self.time_start = time.time()
                self.conn()
                buf = Lib.pack16bit(len(self.request)) + self.request
                self.s.setblocking(0)
                self.s.sendall(buf)
                r = self.processTCPReply()
                if r.header['id'] == self.tid:
                    self.response = r
                    break
            finally:
                self.s.close()
        except socket.error as e:
            first_socket_error = first_socket_error or e
            continue
        except TimeoutError as t:
            first_socket_error = first_socket_error or t
            continue
        if self.response:
            break
    if not self.response and first_socket_error:
        raise first_socket_error