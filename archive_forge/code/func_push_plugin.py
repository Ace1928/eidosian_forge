from .. import auth, utils
@utils.minimum_version('1.25')
@utils.check_resource('name')
def push_plugin(self, name):
    """
            Push a plugin to the registry.

            Args:
                name (string): Name of the plugin to upload. The ``:latest``
                    tag is optional, and is the default if omitted.

            Returns:
                ``True`` if successful
        """
    url = self._url('/plugins/{0}/pull', name)
    headers = {}
    registry, repo_name = auth.resolve_repository_name(name)
    header = auth.get_config_header(self, registry)
    if header:
        headers['X-Registry-Auth'] = header
    res = self._post(url, headers=headers)
    self._raise_for_status(res)
    return self._stream_helper(res, decode=True)