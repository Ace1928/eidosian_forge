from keystoneauth1 import _utils as utils
def add_microversion(self, href, id, **kwargs):
    """Add a microversion version to the list.

        The parameters are the same as MicroversionDiscovery.
        """
    obj = MicroversionDiscovery(href=href, id=id, **kwargs)
    self.add_version(obj)
    return obj