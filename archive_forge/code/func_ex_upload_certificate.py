from libcloud.utils.py3 import u
from libcloud.utils.xml import findall, findattr, findtext
from libcloud.utils.misc import ReprMixin
from libcloud.common.types import LibcloudError
from libcloud.common.aliyun import AliyunXmlResponse, SignedAliyunConnection
from libcloud.loadbalancer.base import Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State
def ex_upload_certificate(self, name, server_certificate, private_key):
    """
        Upload certificate and private key for https load balancer listener

        :param name: the certificate name
        :type name: ``str``

        :param server_certificate: the content of the certificate to upload
                                   in PEM format
        :type server_certificate: ``str``

        :param private_key: the content of the private key to upload
                            in PEM format
        :type private_key: ``str``

        :return: new created certificate info
        :rtype: ``SLBServerCertificate``
        """
    params = {'Action': 'UploadServerCertificate', 'RegionId': self.region, 'ServerCertificate': server_certificate, 'PrivateKey': private_key}
    if name:
        params['ServerCertificateName'] = name
    resp_body = self.connection.request(self.path, params).object
    return self._to_server_certificate(resp_body)