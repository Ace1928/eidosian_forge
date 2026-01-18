import hashlib
import logging
import os
import shutil
import subprocess
import tempfile
from . import DistlibException
from .compat import (HTTPBasicAuthHandler, Request, HTTPPasswordMgr,
from .util import zip_dir, ServerProxy
def get_sign_command(self, filename, signer, sign_password, keystore=None):
    """
        Return a suitable command for signing a file.

        :param filename: The pathname to the file to be signed.
        :param signer: The identifier of the signer of the file.
        :param sign_password: The passphrase for the signer's
                              private key used for signing.
        :param keystore: The path to a directory which contains the keys
                         used in verification. If not specified, the
                         instance's ``gpg_home`` attribute is used instead.
        :return: The signing command as a list suitable to be
                 passed to :class:`subprocess.Popen`.
        """
    cmd = [self.gpg, '--status-fd', '2', '--no-tty']
    if keystore is None:
        keystore = self.gpg_home
    if keystore:
        cmd.extend(['--homedir', keystore])
    if sign_password is not None:
        cmd.extend(['--batch', '--passphrase-fd', '0'])
    td = tempfile.mkdtemp()
    sf = os.path.join(td, os.path.basename(filename) + '.asc')
    cmd.extend(['--detach-sign', '--armor', '--local-user', signer, '--output', sf, filename])
    logger.debug('invoking: %s', ' '.join(cmd))
    return (cmd, sf)