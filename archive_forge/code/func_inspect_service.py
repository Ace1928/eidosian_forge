from .. import auth, errors, utils
from ..types import ServiceMode
@utils.minimum_version('1.24')
@utils.check_resource('service')
def inspect_service(self, service, insert_defaults=None):
    """
        Return information about a service.

        Args:
            service (str): Service name or ID.
            insert_defaults (boolean): If true, default values will be merged
                into the service inspect output.

        Returns:
            (dict): A dictionary of the server-side representation of the
                service, including all relevant properties.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
    url = self._url('/services/{0}', service)
    params = {}
    if insert_defaults is not None:
        if utils.version_lt(self._version, '1.29'):
            raise errors.InvalidVersion('insert_defaults is not supported in API version < 1.29')
        params['insertDefaults'] = insert_defaults
    return self._result(self._get(url, params=params), True)