from __future__ import absolute_import, division, print_function
import atexit
import base64
import datetime
import traceback
from os.path import isfile
from socket import create_connection, setdefaulttimeout, socket
from ssl import get_server_certificate, DER_cert_to_PEM_cert, CERT_NONE, CERT_REQUIRED
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_bytes
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible_collections.community.crypto.plugins.module_utils.crypto.cryptography_support import (
def send_starttls_packet(sock, server_type):
    if server_type == 'mysql':
        ssl_request_packet = b' \x00\x00\x01\x85\xae\x7f\x00' + b'\x00\x00\x00\x01!\x00\x00\x00' + b'\x00\x00\x00\x00\x00\x00\x00\x00' + b'\x00\x00\x00\x00\x00\x00\x00\x00' + b'\x00\x00\x00\x00'
        sock.recv(8192)
        sock.send(ssl_request_packet)