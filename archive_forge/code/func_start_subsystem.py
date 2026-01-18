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
def start_subsystem(self, name, transport, channel):
    self.sock = channel
    self._log(DEBUG, 'Started sftp server on channel {!r}'.format(channel))
    self._send_server_version()
    self.server.session_started()
    while True:
        try:
            t, data = self._read_packet()
        except EOFError:
            self._log(DEBUG, 'EOF -- end of session')
            return
        except Exception as e:
            self._log(DEBUG, 'Exception on channel: ' + str(e))
            self._log(DEBUG, util.tb_strings())
            return
        msg = Message(data)
        request_number = msg.get_int()
        try:
            self._process(t, request_number, msg)
        except Exception as e:
            self._log(DEBUG, 'Exception in server processing: ' + str(e))
            self._log(DEBUG, util.tb_strings())
            try:
                self._send_status(request_number, SFTP_FAILURE)
            except:
                pass