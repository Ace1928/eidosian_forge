from .. import auth, utils
@utils.minimum_version('1.25')
def inspect_plugin(self, name):
    """
            Retrieve plugin metadata.

            Args:
                name (string): The name of the plugin. The ``:latest`` tag is
                    optional, and is the default if omitted.

            Returns:
                A dict containing plugin info
        """
    url = self._url('/plugins/{0}/json', name)
    return self._result(self._get(url), True)