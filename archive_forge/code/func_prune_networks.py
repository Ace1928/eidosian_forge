from ..errors import InvalidVersion
from ..utils import check_resource, minimum_version
from ..utils import version_lt
from .. import utils
@minimum_version('1.25')
def prune_networks(self, filters=None):
    """
        Delete unused networks

        Args:
            filters (dict): Filters to process on the prune list.

        Returns:
            (dict): A dict containing a list of deleted network names and
                the amount of disk space reclaimed in bytes.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
    params = {}
    if filters:
        params['filters'] = utils.convert_filters(filters)
    url = self._url('/networks/prune')
    return self._result(self._post(url, params=params), True)