from ..transport import Transport
from . import test_sftp_transport
def reset_connections(self):
    self.connections = []