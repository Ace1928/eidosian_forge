from libcloud.utils.xml import findall, findtext
from libcloud.common.aws import AWSGenericResponse, SignedAWSConnection
from libcloud.loadbalancer.base import Driver, Member, LoadBalancer
from libcloud.loadbalancer.types import State
def ex_register_targets(self, target_group, members=None):
    """
        Register members as targets at target group

        :param target_group: Target group dict where register members.
        :type target_group: ``dict``

        :param members: List of Members to attach to the balancer. If 'port'
                        attribute is set for the member - load balancer will
                        send traffic there. Otherwise - load balancer port is
                        used on the memeber's side. 'ip' attribute is ignored.
        :type members: ``list`` of :class:`Member`

        :return: True on success, False if no members provided.
        :rtype: ``bool``
        """
    members = members or []
    params = {'Action': 'RegisterTargets', 'TargetGroupArn': target_group.id}
    if not members:
        return False
    idx = 0
    for member in members:
        idx += 1
        params['Targets.member.' + str(idx) + '.Id'] = member.id
        if member.port:
            params['Targets.member.' + str(idx) + '.Port'] = member.port
    self.connection.request(ROOT, params=params)
    target_group.members = members
    return True