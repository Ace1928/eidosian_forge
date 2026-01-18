from __future__ import absolute_import, division, print_function
import abc
from ansible.module_utils import six
from ansible_collections.community.crypto.plugins.module_utils.acme.errors import (
from ansible_collections.community.crypto.plugins.module_utils.acme.utils import (
from ansible_collections.community.crypto.plugins.module_utils.crypto.pem import (
def retrieve_acme_v1_certificate(client, csr_der):
    """
    Create a new certificate based on the CSR (ACME v1 protocol).
    Return the certificate object as dict
    https://tools.ietf.org/html/draft-ietf-acme-acme-02#section-6.5
    """
    new_cert = {'resource': 'new-cert', 'csr': nopad_b64(csr_der)}
    result, info = client.send_signed_request(client.directory['new-cert'], new_cert, error_msg='Failed to receive certificate', expected_status_codes=[200, 201])
    cert = CertificateChain(info['location'])
    cert.cert = der_to_pem(result)

    def f(link, relation):
        if relation == 'up':
            chain_result, chain_info = client.get_request(link, parse_json_result=False)
            if chain_info['status'] in [200, 201]:
                del cert.chain[:]
                cert.chain.append(der_to_pem(chain_result))
    process_links(info, f)
    return cert