from troveclient import base
def list_datastore_version_associated_flavors(self, datastore, version_id):
    """Get a list of all flavors for the specified datastore type
        and datastore version .
        :rtype: list of :class:`Flavor`.
        """
    return self._list('/datastores/%s/versions/%s/flavors' % (datastore, version_id), 'flavors')