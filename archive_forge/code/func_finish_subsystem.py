import os
import errno
import sys
from hashlib import md5, sha1
from paramiko import util
from paramiko.sftp import (
from paramiko.sftp_si import SFTPServerInterface
from paramiko.sftp_attr import SFTPAttributes
from paramiko.common import DEBUG
from paramiko.server import SubsystemHandler
from paramiko.util import b
from paramiko.sftp import (
from paramiko.sftp_handle import SFTPHandle
def finish_subsystem(self):
    self.server.session_ended()
    super().finish_subsystem()
    for f in self.file_table.values():
        f.close()
    for f in self.folder_table.values():
        f.close()
    self.file_table = {}
    self.folder_table = {}