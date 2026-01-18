import hashlib
import logging
import os
import shutil
import subprocess
import tempfile
from . import DistlibException
from .compat import (HTTPBasicAuthHandler, Request, HTTPPasswordMgr,
from .util import zip_dir, ServerProxy
def sign_file(self, filename, signer, sign_password, keystore=None):
    """
        Sign a file.

        :param filename: The pathname to the file to be signed.
        :param signer: The identifier of the signer of the file.
        :param sign_password: The passphrase for the signer's
                              private key used for signing.
        :param keystore: The path to a directory which contains the keys
                         used in signing. If not specified, the instance's
                         ``gpg_home`` attribute is used instead.
        :return: The absolute pathname of the file where the signature is
                 stored.
        """
    cmd, sig_file = self.get_sign_command(filename, signer, sign_password, keystore)
    rc, stdout, stderr = self.run_command(cmd, sign_password.encode('utf-8'))
    if rc != 0:
        raise DistlibException('sign command failed with error code %s' % rc)
    return sig_file