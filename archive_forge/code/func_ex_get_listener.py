from libcloud.utils.xml import findall, findtext
from libcloud.common.aws import AWSGenericResponse, SignedAWSConnection
from libcloud.loadbalancer.base import Driver, Member, LoadBalancer
from libcloud.loadbalancer.types import State
def ex_get_listener(self, listener_id):
    """
        Get listener object by ARN

        :param listener_id: ARN of listener object to get
        :type listener_id: ``str``

        :return: Listener object
        :rtype: :class:`ALBListener`
        """
    params = {'Action': 'DescribeListeners', 'ListenerArns.member.1': listener_id}
    data = self.connection.request(ROOT, params=params).object
    return self._to_listeners(data)[0]