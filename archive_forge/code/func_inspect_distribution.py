import logging
import os
from .. import auth, errors, utils
from ..constants import DEFAULT_DATA_CHUNK_SIZE
@utils.minimum_version('1.30')
@utils.check_resource('image')
def inspect_distribution(self, image, auth_config=None):
    """
        Get image digest and platform information by contacting the registry.

        Args:
            image (str): The image name to inspect
            auth_config (dict): Override the credentials that are found in the
                config for this request.  ``auth_config`` should contain the
                ``username`` and ``password`` keys to be valid.

        Returns:
            (dict): A dict containing distribution data

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
    registry, _ = auth.resolve_repository_name(image)
    headers = {}
    if auth_config is None:
        header = auth.get_config_header(self, registry)
        if header:
            headers['X-Registry-Auth'] = header
    else:
        log.debug('Sending supplied auth config')
        headers['X-Registry-Auth'] = auth.encode_header(auth_config)
    url = self._url('/distribution/{0}/json', image)
    return self._result(self._get(url, headers=headers), True)