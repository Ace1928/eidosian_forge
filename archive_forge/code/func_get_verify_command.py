import hashlib
import logging
import os
import shutil
import subprocess
import tempfile
from . import DistlibException
from .compat import (HTTPBasicAuthHandler, Request, HTTPPasswordMgr,
from .util import zip_dir, ServerProxy
def get_verify_command(self, signature_filename, data_filename, keystore=None):
    """
        Return a suitable command for verifying a file.

        :param signature_filename: The pathname to the file containing the
                                   signature.
        :param data_filename: The pathname to the file containing the
                              signed data.
        :param keystore: The path to a directory which contains the keys
                         used in verification. If not specified, the
                         instance's ``gpg_home`` attribute is used instead.
        :return: The verifying command as a list suitable to be
                 passed to :class:`subprocess.Popen`.
        """
    cmd = [self.gpg, '--status-fd', '2', '--no-tty']
    if keystore is None:
        keystore = self.gpg_home
    if keystore:
        cmd.extend(['--homedir', keystore])
    cmd.extend(['--verify', signature_filename, data_filename])
    logger.debug('invoking: %s', ' '.join(cmd))
    return cmd