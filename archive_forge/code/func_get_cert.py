import ctypes
import json
import logging
import os
import sys
import cffi  # type: ignore
import six
from google.auth import exceptions
def get_cert(signer_lib, config_file_path):
    cert_len = signer_lib.GetCertPemForPython(config_file_path.encode(), None, 0)
    if cert_len == 0:
        raise exceptions.MutualTLSChannelError('failed to get certificate')
    cert_holder = ctypes.create_string_buffer(cert_len)
    signer_lib.GetCertPemForPython(config_file_path.encode(), cert_holder, cert_len)
    return bytes(cert_holder)