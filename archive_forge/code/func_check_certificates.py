import os
import re
import base64
import collections
from libcloud.utils.py3 import b, httplib
from libcloud.common.base import JsonResponse, ConnectionUserAndKey, KeyCertificateConnection
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import StorageVolume
from libcloud.container.base import Container, ContainerImage, ContainerDriver
from libcloud.container.types import ContainerState
from libcloud.common.exceptions import BaseHTTPError
from libcloud.container.providers import Provider
def check_certificates(key_file, cert_file, **kwargs):
    """
    Basic checks for the provided certificates in LXDtlsConnection
    """
    if key_file is None or cert_file is None:
        raise InvalidCredsError('TLS Connection requires specification of a key file and a certificate file')
    if key_file == '' or cert_file == '':
        raise InvalidCredsError('TLS Connection requires specification of a key file and a certificate file')
    if 'key_files_allowed' in kwargs.keys():
        key_file_suffix = key_file.split('.')
        if key_file_suffix[-1] not in kwargs['key_files_allowed']:
            raise InvalidCredsError('Valid key files are: ' + str(kwargs['key_files_allowed']) + 'you provided: ' + key_file_suffix[-1])
    if 'cert_files_allowed' in kwargs.keys():
        cert_file_suffix = cert_file.split('.')
        if cert_file_suffix[-1] not in kwargs['cert_files_allowed']:
            raise InvalidCredsError('Valid certification files are: ' + str(kwargs['cert_files_allowed']) + 'you provided: ' + cert_file_suffix[-1])
    keypath = os.path.expanduser(key_file)
    is_file_path = os.path.exists(keypath) and os.path.isfile(keypath)
    if not is_file_path:
        raise InvalidCredsError('You need a key file to authenticate with LXD tls. This can be found in the server.')
    certpath = os.path.expanduser(cert_file)
    is_file_path = os.path.exists(certpath) and os.path.isfile(certpath)
    if not is_file_path:
        raise InvalidCredsError('You need a certificate file to authenticate with LXD tls. This can be found in the server.')