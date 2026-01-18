from libcloud.utils.xml import findall, findtext
from libcloud.common.aws import AWSGenericResponse, SignedAWSConnection
from libcloud.loadbalancer.base import Driver, Member, LoadBalancer
from libcloud.loadbalancer.types import State
def ex_create_listener(self, balancer, port, proto, target_group, action='forward', ssl_cert_arn=None, ssl_policy=None):
    """
        Create a listener for application load balancer

        :param balancer: LoadBalancer to create listener for
        :type  balancer: :class:`LoadBalancer`

        :param port: Port number to setup load balancer listener
        :type port: ``int``

        :param proto: Load balancer protocol, should be 'HTTP' or 'HTTPS'.
        :type proto: ``str``

        :param target_group: Target group associated with the listener.
        :type target_group: :class:`ALBTargetGroup`

        :param action: Default action for the listener,
                        valid value is 'forward'
        :type action: ``str``

        :param ssl_cert_arn: SSL certificate ARN to use when listener protocol
                            is 'HTTPS'.
        :type ssl_cert_arn: ``str``

        :param ssl_policy: The security policy that defines which ciphers and
                        protocols are supported. The default is the current
                        predefined security policy.
                        Example: 'ELBSecurityPolicy-2016-08'
        :type ssl_policy: ``str``

        :return: Listener object
        :rtype: :class:`ALBListener`
        """
    ssl_cert_arn = ssl_cert_arn or ''
    ssl_policy = ssl_policy or ''
    params = {'Action': 'CreateListener', 'LoadBalancerArn': balancer.id, 'Protocol': proto, 'Port': port, 'DefaultActions.member.1.Type': action, 'DefaultActions.member.1.TargetGroupArn': target_group.id}
    if proto == 'HTTPS':
        params['Certificates.member.1.CertificateArn'] = ssl_cert_arn
        if ssl_policy:
            params['SslPolicy'] = ssl_policy
    data = self.connection.request(ROOT, params=params).object
    xpath = 'CreateListenerResult/Listeners/member'
    for el in findall(element=data, xpath=xpath, namespace=NS):
        listener = self._to_listener(el)
        listener.balancer = balancer
    return listener