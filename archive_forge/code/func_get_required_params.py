from libcloud.utils.py3 import u
from libcloud.utils.xml import findall, findattr, findtext
from libcloud.utils.misc import ReprMixin
from libcloud.common.types import LibcloudError
from libcloud.common.aliyun import AliyunXmlResponse, SignedAliyunConnection
from libcloud.loadbalancer.base import Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State
def get_required_params(self):
    params = super().get_required_params()
    params['StickySession'] = self.sticky_session
    params['HealthCheck'] = self.health_check
    params['ServerCertificateId'] = self.certificate_id
    return params