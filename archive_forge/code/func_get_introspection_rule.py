from openstack import _log
from openstack.baremetal.v1 import node as _node
from openstack.baremetal_introspection.v1 import introspection as _introspect
from openstack.baremetal_introspection.v1 import (
from openstack import exceptions
from openstack import proxy
def get_introspection_rule(self, introspection_rule):
    """Get a specific introspection rule.

        :param introspection_rule: The value can be the name or ID of an
            introspection rule or a
            :class:`~.introspection_rule.IntrospectionRule` instance.

        :returns: :class:`~.introspection_rule.IntrospectionRule` instance.
        :raises: :class:`~openstack.exceptions.ResourceNotFound` when no
            introspection rule matching the name or ID could be found.
        """
    return self._get(_introspection_rule.IntrospectionRule, introspection_rule)