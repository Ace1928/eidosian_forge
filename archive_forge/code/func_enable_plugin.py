from .. import auth, utils
@utils.minimum_version('1.25')
def enable_plugin(self, name, timeout=0):
    """
            Enable an installed plugin.

            Args:
                name (string): The name of the plugin. The ``:latest`` tag is
                    optional, and is the default if omitted.
                timeout (int): Operation timeout (in seconds). Default: 0

            Returns:
                ``True`` if successful
        """
    url = self._url('/plugins/{0}/enable', name)
    params = {'timeout': timeout}
    res = self._post(url, params=params)
    self._raise_for_status(res)
    return True