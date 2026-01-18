from openstack.accelerator.v2._proxy import Proxy
def list_device_profiles(self, filters=None):
    """List all device_profiles.

        :param filters: (optional) dict of filter conditions to push down
        :returns: A list of accelerator ``DeviceProfile`` objects.
        """
    if not filters:
        filters = {}
    return list(self.accelerator.device_profiles(**filters))