import optparse
import os
import sys
from subprocess import PIPE, CalledProcessError, Popen
from breezy import osutils
from breezy.tests import ssl_certs
def sign_server_certificate():
    """CA signs server csr"""
    server_csr_path = ssl_certs.build_path('server.csr')
    ca_cert_path = ssl_certs.build_path('ca.crt')
    ca_key_path = ssl_certs.build_path('ca.key')
    needs('Signing server.crt', server_csr_path, ca_cert_path, ca_key_path)
    server_cert_path = ssl_certs.build_path('server.crt')
    server_ext_conf = ssl_certs.build_path('server.extensions.cnf')
    rm_f(server_cert_path)
    _openssl(['x509', '-req', '-passin', 'stdin', '-days', '365242', '-in', server_csr_path, '-CA', ca_cert_path, '-CAkey', ca_key_path, '-set_serial', '01', '-extfile', server_ext_conf, '-out', server_cert_path], input='%(ca_pass)s\n' % ssl_params)