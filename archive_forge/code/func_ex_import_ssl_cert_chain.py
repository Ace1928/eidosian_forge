from libcloud.utils.py3 import ET
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.utils.misc import reverse_dict
from libcloud.common.nttcis import (
from libcloud.loadbalancer.base import DEFAULT_ALGORITHM, Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State, Provider
def ex_import_ssl_cert_chain(self, network_domain_id, name, chain_crt_file, description=None):
    """
        Import an ssl certificate chain for ssl offloading onto
        the the load balancer

        :param network_domain_id:  The Network Domain's Id.
        :type network_domain_id: ``str``
        :param name: The name of the ssl certificate chain
        :type name: ``str``
        :param chain_crt_file: The complete path to the certificate chain file
        :type chain_crt_file: ``str``
        :param description: (Optional) A description of the certificate chain
        :type description: ``str``
        :return: ``bool``
        """
    try:
        import OpenSSL
        from OpenSSL import crypto
    except ImportError:
        raise ImportError('Missing "OpenSSL" dependency. You can install it using pip - pip install pyopenssl')
    c = crypto.load_certificate(crypto.FILETYPE_PEM, open(chain_crt_file).read())
    cert = OpenSSL.crypto.dump_certificate(crypto.FILETYPE_PEM, c).decode(encoding='utf-8')
    cert_chain_elem = ET.Element('importSslCertificateChain', {'xmlns': TYPES_URN})
    ET.SubElement(cert_chain_elem, 'networkDomainId').text = network_domain_id
    ET.SubElement(cert_chain_elem, 'name').text = name
    if description is not None:
        ET.SubElement(cert_chain_elem, 'description').text = description
    ET.SubElement(cert_chain_elem, 'certificateChain').text = cert
    result = self.connection.request_with_orgId_api_2('networkDomainVip/importSslCertificateChain', method='POST', data=ET.tostring(cert_chain_elem)).object
    response_code = findtext(result, 'responseCode', TYPES_URN)
    return response_code in ['IN_PROGRESS', 'OK']