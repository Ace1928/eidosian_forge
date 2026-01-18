import base64
from .. import errors
from .. import utils
@utils.minimum_version('1.25')
@utils.check_resource('id')
def inspect_secret(self, id):
    """
            Retrieve secret metadata

            Args:
                id (string): Full ID of the secret to inspect

            Returns (dict): A dictionary of metadata

            Raises:
                :py:class:`docker.errors.NotFound`
                    if no secret with that ID exists
        """
    url = self._url('/secrets/{0}', id)
    return self._result(self._get(url), True)