from binascii import hexlify
import errno
import os
import stat
import threading
import time
import weakref
from paramiko import util
from paramiko.channel import Channel
from paramiko.message import Message
from paramiko.common import INFO, DEBUG, o777
from paramiko.sftp import (
from paramiko.sftp_attr import SFTPAttributes
from paramiko.ssh_exception import SSHException
from paramiko.sftp_file import SFTPFile
from paramiko.util import ClosingContextManager, b, u
def listdir_iter(self, path='.', read_aheads=50):
    """
        Generator version of `.listdir_attr`.

        See the API docs for `.listdir_attr` for overall details.

        This function adds one more kwarg on top of `.listdir_attr`:
        ``read_aheads``, an integer controlling how many
        ``SSH_FXP_READDIR`` requests are made to the server. The default of 50
        should suffice for most file listings as each request/response cycle
        may contain multiple files (dependent on server implementation.)

        .. versionadded:: 1.15
        """
    path = self._adjust_cwd(path)
    self._log(DEBUG, 'listdir({!r})'.format(path))
    t, msg = self._request(CMD_OPENDIR, path)
    if t != CMD_HANDLE:
        raise SFTPError('Expected handle')
    handle = msg.get_string()
    nums = list()
    while True:
        try:
            for i in range(read_aheads):
                num = self._async_request(type(None), CMD_READDIR, handle)
                nums.append(num)
            for num in nums:
                t, pkt_data = self._read_packet()
                msg = Message(pkt_data)
                new_num = msg.get_int()
                if num == new_num:
                    if t == CMD_STATUS:
                        self._convert_status(msg)
                count = msg.get_int()
                for i in range(count):
                    filename = msg.get_text()
                    longname = msg.get_text()
                    attr = SFTPAttributes._from_msg(msg, filename, longname)
                    if filename != '.' and filename != '..':
                        yield attr
            nums = list()
        except EOFError:
            self._request(CMD_CLOSE, handle)
            return