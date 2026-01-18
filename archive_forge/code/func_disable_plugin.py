from .. import auth, utils
@utils.minimum_version('1.25')
def disable_plugin(self, name, force=False):
    """
            Disable an installed plugin.

            Args:
                name (string): The name of the plugin. The ``:latest`` tag is
                    optional, and is the default if omitted.
                force (bool): To enable the force query parameter.

            Returns:
                ``True`` if successful
        """
    url = self._url('/plugins/{0}/disable', name)
    res = self._post(url, params={'force': force})
    self._raise_for_status(res)
    return True