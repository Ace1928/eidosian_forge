from io import BytesIO
from os import SEEK_END
import dulwich
from .errors import GitProtocolError, HangupException
def send_cmd(self, cmd, *args):
    """Send a command and some arguments to a git server.

        Only used for the TCP git protocol (git://).

        Args:
          cmd: The remote service to access.
          args: List of arguments to send to remove service.
        """
    self.write_pkt_line(format_cmd_pkt(cmd, *args))