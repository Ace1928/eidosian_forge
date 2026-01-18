import errno
import socket
from pathlib import Path
from threading import Thread
import zmq
from jupyter_client.localinterfaces import localhost
def pick_port(self):
    """Pick a port for the heartbeat."""
    if self.transport == 'tcp':
        s = socket.socket()
        s.bind(('' if self.ip == '*' else self.ip, 0))
        self.port = s.getsockname()[1]
        s.close()
    elif self.transport == 'ipc':
        self.port = 1
        while Path(f'{self.ip}-{self.port}').exists():
            self.port = self.port + 1
    else:
        raise ValueError('Unrecognized zmq transport: %s' % self.transport)
    return self.port