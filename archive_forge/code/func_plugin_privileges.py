from .. import auth, utils
@utils.minimum_version('1.25')
def plugin_privileges(self, name):
    """
            Retrieve list of privileges to be granted to a plugin.

            Args:
                name (string): Name of the remote plugin to examine. The
                    ``:latest`` tag is optional, and is the default if omitted.

            Returns:
                A list of dictionaries representing the plugin's
                permissions

        """
    params = {'remote': name}
    headers = {}
    registry, repo_name = auth.resolve_repository_name(name)
    header = auth.get_config_header(self, registry)
    if header:
        headers['X-Registry-Auth'] = header
    url = self._url('/plugins/privileges')
    return self._result(self._get(url, params=params, headers=headers), True)