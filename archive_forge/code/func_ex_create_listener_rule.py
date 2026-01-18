from libcloud.utils.xml import findall, findtext
from libcloud.common.aws import AWSGenericResponse, SignedAWSConnection
from libcloud.loadbalancer.base import Driver, Member, LoadBalancer
from libcloud.loadbalancer.types import State
def ex_create_listener_rule(self, listener, priority, target_group, action='forward', condition_field=None, condition_value=None):
    """
        Create a rule for listener.

        :param listener: Listener object where to create rule
        :type listener: :class:`ALBListener`

        :param priority: The priority for the rule. A listener can't have
                        multiple rules with the same priority.
        :type priority: ``str``

        :param target_group: Target group object to associate with rule
        :type target_group: :class:`ALBTargetGroup`

        :param action: Action for the rule, valid value is 'forward'
        :type action: ``str``

        :param condition_field: Rule condition field name. The possible values
                                are 'host-header' and 'path-pattern'.
        :type condition_field: ``str``

        :param condition_value: Value to match. Wildcards are supported, for
                                example: '/img/*'

        :return: Rule object
        :rtype: :class:`ALBRule`
        """
    condition_field = condition_field or ''
    condition_value = condition_value or ''
    params = {'Action': 'CreateRule', 'ListenerArn': listener.id, 'Priority': priority, 'Actions.member.1.Type': action, 'Actions.member.1.TargetGroupArn': target_group.id, 'Conditions.member.1.Field': condition_field, 'Conditions.member.1.Values.member.1': condition_value}
    data = self.connection.request(ROOT, params=params).object
    xpath = 'CreateRuleResult/Rules/member'
    for el in findall(element=data, xpath=xpath, namespace=NS):
        rule = self._to_rule(el)
        rule.listener = listener
    return rule