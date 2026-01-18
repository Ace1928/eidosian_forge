from .. import auth, errors, utils
from ..types import ServiceMode
@utils.minimum_version('1.24')
def tasks(self, filters=None):
    """
        Retrieve a list of tasks.

        Args:
            filters (dict): A map of filters to process on the tasks list.
                Valid filters: ``id``, ``name``, ``service``, ``node``,
                ``label`` and ``desired-state``.

        Returns:
            (:py:class:`list`): List of task dictionaries.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
    params = {'filters': utils.convert_filters(filters) if filters else None}
    url = self._url('/tasks')
    return self._result(self._get(url, params=params), True)