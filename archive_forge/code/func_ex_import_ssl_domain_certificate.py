from libcloud.utils.py3 import ET
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.utils.misc import reverse_dict
from libcloud.common.nttcis import (
from libcloud.loadbalancer.base import DEFAULT_ALGORITHM, Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State, Provider
def ex_import_ssl_domain_certificate(self, network_domain_id, name, crt_file, key_file, description=None):
    """
        Import an ssl cert for ssl offloading onto the the load balancer

        :param network_domain_id:  The Network Domain's Id.
        :type network_domain_id: ``str``
        :param name: The name of the ssl certificate
        :type name: ``str``
        :param crt_file: The complete path to the certificate file
        :type crt_file: ``str``
        :param key_file: The complete pathy to the key file
        :type key_file: ``str``
        :param description: (Optional) A description of the certificate
        :type `description: `str``
        :return: ``bool``
        """
    try:
        import OpenSSL
    except ImportError:
        raise ImportError('Missing "OpenSSL" dependency. You can install it using pip - pip install pyopenssl')
    with open(crt_file) as fp:
        c = OpenSSL.crypto.load_certificate(OpenSSL.crypto.FILETYPE_PEM, fp.read())
    cert = OpenSSL.crypto.dump_certificate(OpenSSL.crypto.FILETYPE_PEM, c).decode(encoding='utf-8')
    with open(key_file) as fp:
        k = OpenSSL.crypto.load_privatekey(OpenSSL.crypto.FILETYPE_PEM, fp.read())
    key = OpenSSL.crypto.dump_privatekey(OpenSSL.crypto.FILETYPE_PEM, k).decode(encoding='utf-8')
    cert_elem = ET.Element('importSslDomainCertificate', {'xmlns': TYPES_URN})
    ET.SubElement(cert_elem, 'networkDomainId').text = network_domain_id
    ET.SubElement(cert_elem, 'name').text = name
    if description is not None:
        ET.SubElement(cert_elem, 'description').text = description
    ET.SubElement(cert_elem, 'key').text = key
    ET.SubElement(cert_elem, 'certificate').text = cert
    result = self.connection.request_with_orgId_api_2('networkDomainVip/importSslDomainCertificate', method='POST', data=ET.tostring(cert_elem)).object
    response_code = findtext(result, 'responseCode', TYPES_URN)
    return response_code in ['IN_PROGRESS', 'OK']