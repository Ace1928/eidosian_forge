import os
import sys
from paramiko.sftp import SFTP_OP_UNSUPPORTED
def session_ended(self):
    """
        The SFTP server session has just ended, either cleanly or via an
        exception.  This method is meant to be overridden to perform any
        necessary cleanup before this `.SFTPServerInterface` object is
        destroyed.
        """
    pass