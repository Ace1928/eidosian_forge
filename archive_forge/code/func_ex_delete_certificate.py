from libcloud.utils.py3 import u
from libcloud.utils.xml import findall, findattr, findtext
from libcloud.utils.misc import ReprMixin
from libcloud.common.types import LibcloudError
from libcloud.common.aliyun import AliyunXmlResponse, SignedAliyunConnection
from libcloud.loadbalancer.base import Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State
def ex_delete_certificate(self, certificate_id):
    """
        Delete the given server certificate

        :param certificate_id: the id of the certificate to delete
        :type certificate_id: ``str``

        :return: whether process is success
        :rtype: ``bool``
        """
    params = {'Action': 'DeleteServerCertificate', 'RegionId': self.region, 'ServerCertificateId': certificate_id}
    resp = self.connection.request(self.path, params)
    return resp.success()