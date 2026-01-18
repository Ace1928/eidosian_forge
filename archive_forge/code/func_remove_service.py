from .. import auth, errors, utils
from ..types import ServiceMode
@utils.minimum_version('1.24')
@utils.check_resource('service')
def remove_service(self, service):
    """
        Stop and remove a service.

        Args:
            service (str): Service name or ID

        Returns:
            ``True`` if successful.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
    url = self._url('/services/{0}', service)
    resp = self._delete(url)
    self._raise_for_status(resp)
    return True