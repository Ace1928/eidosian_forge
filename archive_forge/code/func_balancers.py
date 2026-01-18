from libcloud.utils.xml import findall, findtext
from libcloud.common.aws import AWSGenericResponse, SignedAWSConnection
from libcloud.loadbalancer.base import Driver, Member, LoadBalancer
from libcloud.loadbalancer.types import State
@balancers.setter
def balancers(self, val):
    self._balancers = val
    self._balancers_arns = [lb.id for lb in val] if val else []