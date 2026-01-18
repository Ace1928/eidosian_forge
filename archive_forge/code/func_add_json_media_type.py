from keystoneauth1 import _utils as utils
def add_json_media_type(self):
    """Add the JSON media-type links.

        The standard structure includes a list of media-types that the endpoint
        supports. Add JSON to the list.
        """
    self.add_media_type(base='application/json', type='application/vnd.openstack.identity-v3+json')