from io import BytesIO
from os import SEEK_END
import dulwich
from .errors import GitProtocolError, HangupException
def read_cmd(self):
    """Read a command and some arguments from the git client.

        Only used for the TCP git protocol (git://).

        Returns: A tuple of (command, [list of arguments]).
        """
    line = self.read_pkt_line()
    return parse_cmd_pkt(line)