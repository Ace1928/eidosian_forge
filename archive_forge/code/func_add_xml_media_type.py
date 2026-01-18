from keystoneauth1 import _utils as utils
def add_xml_media_type(self):
    """Add the XML media-type links.

        The standard structure includes a list of media-types that the endpoint
        supports. Add XML to the list.
        """
    self.add_media_type(base='application/xml', type='application/vnd.openstack.identity-v3+xml')