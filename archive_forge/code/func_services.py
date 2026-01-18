from .. import auth, errors, utils
from ..types import ServiceMode
@utils.minimum_version('1.24')
def services(self, filters=None, status=None):
    """
        List services.

        Args:
            filters (dict): Filters to process on the nodes list. Valid
                filters: ``id``, ``name`` , ``label`` and ``mode``.
                Default: ``None``.
            status (bool): Include the service task count of running and
                desired tasks. Default: ``None``.

        Returns:
            A list of dictionaries containing data about each service.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
    params = {'filters': utils.convert_filters(filters) if filters else None}
    if status is not None:
        if utils.version_lt(self._version, '1.41'):
            raise errors.InvalidVersion('status is not supported in API version < 1.41')
        params['status'] = status
    url = self._url('/services')
    return self._result(self._get(url, params=params), True)