from openstack import exceptions
from openstack import resource
from openstack import utils
def delete_metadata_item(self, session, key):
    """Removes a single metadata item from the specified resource.

        :param session: The session to use for making this request.
        :param str key: The key as a string.
        """
    url = utils.urljoin(self.base_path, self.id, 'metadata', key)
    response = session.delete(url)
    exceptions.raise_from_response(response)
    metadata = self.metadata
    try:
        if metadata:
            metadata.pop(key)
        else:
            metadata = {}
    except ValueError:
        pass
    self._body.attributes.update({'metadata': metadata})
    return self