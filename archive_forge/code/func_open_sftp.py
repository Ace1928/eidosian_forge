from boto.mashups.interactive import interactive_shell
import boto
import os
import time
import shutil
import paramiko
import socket
import subprocess
from boto.compat import StringIO
def open_sftp(self):
    """
        Open an SFTP session on the SSH server.
        
        :rtype: :class:`paramiko.sftp_client.SFTPClient`
        :return: An SFTP client object.
        """
    return self._ssh_client.open_sftp()