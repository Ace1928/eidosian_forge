import optparse
import os
import sys
from subprocess import PIPE, CalledProcessError, Popen
from breezy import osutils
from breezy.tests import ssl_certs
def build_server_signing_request():
    """Create a CSR (certificate signing request) to get signed by the CA"""
    key_path = ssl_certs.build_path('server_with_pass.key')
    needs('Building server.csr', key_path)
    server_csr_path = ssl_certs.build_path('server.csr')
    rm_f(server_csr_path)
    _openssl(['req', '-passin', 'stdin', '-new', '-key', key_path, '-out', server_csr_path], input='%(server_pass)s\n%(server_country_code)s\n%(server_state)s\n%(server_locality)s\n%(server_organization)s\n%(server_section)s\n%(server_name)s\n%(server_email)s\n%(server_challenge_pass)s\n%(server_optional_company_name)s\n' % ssl_params)