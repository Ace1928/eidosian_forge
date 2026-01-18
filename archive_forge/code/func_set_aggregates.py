from openstack import exceptions
from openstack import resource
from openstack import utils
def set_aggregates(self, session, aggregates=None):
    """Replaces aggregates on the resource provider

        :param session: The session to use for making this request
        :param list aggregates: List of aggregates
        :return: The resource provider with updated aggregates populated
        """
    url = utils.urljoin(self.base_path, self.id, 'aggregates')
    microversion = self._get_microversion(session, action='commit')
    body = {'aggregates': aggregates or []}
    if utils.supports_microversion(session, '1.19'):
        body['resource_provider_generation'] = self.generation
    response = session.put(url, json=body, microversion=microversion)
    exceptions.raise_from_response(response)
    data = response.json()
    updates = {'aggregates': data['aggregates']}
    if 'resource_provider_generation' in data:
        updates['resource_provider_generation'] = data['resource_provider_generation']
    self._body.attributes.update(updates)
    return self