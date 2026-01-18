from openstack.accelerator.v2._proxy import Proxy
def list_accelerator_requests(self, filters=None):
    """List all accelerator_requests.

        :param filters: (optional) dict of filter conditions to push down
        :returns: A list of accelerator ``AcceleratorRequest`` objects.
        """
    if not filters:
        filters = {}
    return list(self.accelerator.accelerator_requests(**filters))