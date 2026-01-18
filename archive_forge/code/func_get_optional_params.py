from libcloud.utils.py3 import u
from libcloud.utils.xml import findall, findattr, findtext
from libcloud.utils.misc import ReprMixin
from libcloud.common.types import LibcloudError
from libcloud.common.aliyun import AliyunXmlResponse, SignedAliyunConnection
from libcloud.loadbalancer.base import Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State
def get_optional_params(self):
    options = {}
    for option in self.option_keys:
        if self.extra and option in self.extra:
            options[option] = self.extra[option]
    return options