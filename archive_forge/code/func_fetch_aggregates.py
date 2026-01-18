from openstack import exceptions
from openstack import resource
from openstack import utils
def fetch_aggregates(self, session):
    """List aggregates set on the resource provider

        :param session: The session to use for making this request
        :return: The resource provider with aggregates populated
        """
    url = utils.urljoin(self.base_path, self.id, 'aggregates')
    microversion = self._get_microversion(session, action='fetch')
    response = session.get(url, microversion=microversion)
    exceptions.raise_from_response(response)
    data = response.json()
    updates = {'aggregates': data['aggregates']}
    if utils.supports_microversion(session, '1.19'):
        updates['generation'] = data['resource_provider_generation']
    self._body.attributes.update(updates)
    return self