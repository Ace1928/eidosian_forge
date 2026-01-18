from openstack import exceptions
from openstack import resource
from openstack import utils
def fetch_extra_specs(self, session):
    """Fetch extra specs of the flavor

        Starting with 2.61 extra specs are returned with the flavor details,
        before that a separate call is required.

        :param session: The session to use for making this request.
        :returns: The updated flavor.
        """
    url = utils.urljoin(Flavor.base_path, self.id, 'os-extra_specs')
    microversion = self._get_microversion(session, action='fetch')
    response = session.get(url, microversion=microversion)
    exceptions.raise_from_response(response)
    specs = response.json().get('extra_specs', {})
    self._update(extra_specs=specs)
    return self