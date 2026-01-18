from openstack.accelerator.v2._proxy import Proxy
def list_deployables(self, filters=None):
    """List all available deployables.

        :param filters: (optional) dict of filter conditions to push down
        :returns: A list of accelerator ``Deployable`` objects.
        """
    if not filters:
        filters = {}
    return list(self.accelerator.deployables(**filters))