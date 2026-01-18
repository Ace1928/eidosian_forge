from ..transport import Transport
from . import test_sftp_transport
def start_logging_connections(self):
    Transport.hooks.install_named_hook('post_connect', self.connections.append, None)