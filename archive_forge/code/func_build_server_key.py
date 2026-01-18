import optparse
import os
import sys
from subprocess import PIPE, CalledProcessError, Popen
from breezy import osutils
from breezy.tests import ssl_certs
def build_server_key():
    """Generate an ssl server private key.

    We generates a key with a password and then copy it without password so
    that a server can use it without prompting.
    """
    key_path = ssl_certs.build_path('server_with_pass.key')
    rm_f(key_path)
    _openssl(['genrsa', '-passout', 'stdin', '-des3', '-out', key_path, '4096'], input='%(server_pass)s\n%(server_pass)s\n' % ssl_params)
    key_nopass_path = ssl_certs.build_path('server_without_pass.key')
    rm_f(key_nopass_path)
    _openssl(['rsa', '-passin', 'stdin', '-in', key_path, '-out', key_nopass_path], input='%(server_pass)s\n' % ssl_params)