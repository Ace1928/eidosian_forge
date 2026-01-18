from libcloud.utils.py3 import u
from libcloud.utils.xml import findall, findattr, findtext
from libcloud.utils.misc import ReprMixin
from libcloud.common.types import LibcloudError
from libcloud.common.aliyun import AliyunXmlResponse, SignedAliyunConnection
from libcloud.loadbalancer.base import Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State
def ex_list_certificates(self, certificate_ids=[]):
    """
        List all server certificates

        :param certificate_ids: certificate ids to filter results
        :type certificate_ids: ``str``

        :return: certificates
        :rtype: ``SLBServerCertificate``
        """
    params = {'Action': 'DescribeServerCertificates', 'RegionId': self.region}
    if certificate_ids and isinstance(certificate_ids, list):
        params['ServerCertificateId'] = ','.join(certificate_ids)
    resp_body = self.connection.request(self.path, params).object
    cert_elements = findall(resp_body, 'ServerCertificates/ServerCertificate', namespace=self.namespace)
    certificates = [self._to_server_certificate(el) for el in cert_elements]
    return certificates