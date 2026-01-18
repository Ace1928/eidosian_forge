from openstack import exceptions
from openstack.instance_ha.v1 import host as _host
from openstack.instance_ha.v1 import notification as _notification
from openstack.instance_ha.v1 import segment as _segment
from openstack.instance_ha.v1 import vmove as _vmove
from openstack import proxy
from openstack import resource
def create_notification(self, **attrs):
    """Create a new notification.

        :param dict attrs: Keyword arguments which will be used to create
            a :class:`masakariclient.sdk.ha.v1.notification.Notification`,
            comprised of the propoerties on the Notification class.
        :returns: The result of notification creation
        :rtype: :class:`masakariclient.sdk.ha.v1.notification.Notification`
        """
    return self._create(_notification.Notification, **attrs)