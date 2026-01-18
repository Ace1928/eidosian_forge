from .. import errors
from .. import utils
def inspect_volume(self, name):
    """
        Retrieve volume info by name.

        Args:
            name (str): volume name

        Returns:
            (dict): Volume information dictionary

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.

        Example:

            >>> client.api.inspect_volume('foobar')
            {u'Driver': u'local',
             u'Mountpoint': u'/var/lib/docker/volumes/foobar/_data',
             u'Name': u'foobar'}

        """
    url = self._url('/volumes/{0}', name)
    return self._result(self._get(url), True)