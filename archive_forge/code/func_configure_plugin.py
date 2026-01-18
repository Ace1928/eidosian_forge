from .. import auth, utils
@utils.minimum_version('1.25')
@utils.check_resource('name')
def configure_plugin(self, name, options):
    """
            Configure a plugin.

            Args:
                name (string): The name of the plugin. The ``:latest`` tag is
                    optional, and is the default if omitted.
                options (dict): A key-value mapping of options

            Returns:
                ``True`` if successful
        """
    url = self._url('/plugins/{0}/set', name)
    data = options
    if isinstance(data, dict):
        data = [f'{k}={v}' for k, v in data.items()]
    res = self._post_json(url, data=data)
    self._raise_for_status(res)
    return True