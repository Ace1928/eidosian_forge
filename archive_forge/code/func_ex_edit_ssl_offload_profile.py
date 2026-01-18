from libcloud.utils.py3 import ET
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.utils.misc import reverse_dict
from libcloud.common.nttcis import (
from libcloud.loadbalancer.base import DEFAULT_ALGORITHM, Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State, Provider
def ex_edit_ssl_offload_profile(self, profile_id, name, ssl_domain_cert_id, description=None, ciphers=None, ssl_cert_chain_id=None):
    """
        The function edits the ssl offload profile

        :param profil_id: The id of the profile to be edited
        :type profile_id: ``str``
        :param name: The name of the profile, new name or previous name
        :type name: ``str``
        :param ssl_domain_cert_id: The certificate id to use, new or current
        :type ssl_domain_cert_id: ``str``
        :param description: (Optional) Profile's description
        :type description: ``str``
        :param ciphers: (Optional) String of acceptable ciphers to use
        :type ciphers: ``str``
        :param ssl_cert_chain_id: If using a certificate chain
        or changing to a new one
        :type: ssl_cert_chain_id: ``str``
        :returns: ``bool``
        """
    ssl_offload_elem = ET.Element('editSslOffloadProfile', {'xmlns': TYPES_URN, 'id': profile_id})
    ET.SubElement(ssl_offload_elem, 'name').text = name
    if description is not None:
        ET.SubElement(ssl_offload_elem, 'description').text = description
    if ciphers is not None:
        ET.SubElement(ssl_offload_elem, 'ciphers').text = ciphers
    ET.SubElement(ssl_offload_elem, 'sslDomainCertificateId').text = ssl_domain_cert_id
    if ssl_cert_chain_id is not None:
        ET.SubElement(ssl_offload_elem, 'sslCertificateChainId').text = ssl_cert_chain_id
    result = self.connection.request_with_orgId_api_2('networkDomainVip/editSslOffloadProfile', method='POST', data=ET.tostring(ssl_offload_elem)).object
    response_code = findtext(result, 'responseCode', TYPES_URN)
    return response_code in ['IN_PROGRESS', 'OK']