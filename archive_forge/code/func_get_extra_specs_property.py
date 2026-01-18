from openstack import exceptions
from openstack import resource
from openstack import utils
def get_extra_specs_property(self, session, prop):
    """Get an individual extra spec property.

        :param session: The session to use for making this request.
        :param prop: The property to fetch.
        :returns: The value of the property if it exists, else ``None``.
        """
    url = utils.urljoin(Flavor.base_path, self.id, 'os-extra_specs', prop)
    microversion = self._get_microversion(session, action='fetch')
    response = session.get(url, microversion=microversion)
    exceptions.raise_from_response(response)
    val = response.json().get(prop)
    return val