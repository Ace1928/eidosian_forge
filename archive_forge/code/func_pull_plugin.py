from .. import auth, utils
@utils.minimum_version('1.25')
def pull_plugin(self, remote, privileges, name=None):
    """
            Pull and install a plugin. After the plugin is installed, it can be
            enabled using :py:meth:`~enable_plugin`.

            Args:
                remote (string): Remote reference for the plugin to install.
                    The ``:latest`` tag is optional, and is the default if
                    omitted.
                privileges (:py:class:`list`): A list of privileges the user
                    consents to grant to the plugin. Can be retrieved using
                    :py:meth:`~plugin_privileges`.
                name (string): Local name for the pulled plugin. The
                    ``:latest`` tag is optional, and is the default if omitted.

            Returns:
                An iterable object streaming the decoded API logs
        """
    url = self._url('/plugins/pull')
    params = {'remote': remote}
    if name:
        params['name'] = name
    headers = {}
    registry, repo_name = auth.resolve_repository_name(remote)
    header = auth.get_config_header(self, registry)
    if header:
        headers['X-Registry-Auth'] = header
    response = self._post_json(url, params=params, headers=headers, data=privileges, stream=True)
    self._raise_for_status(response)
    return self._stream_helper(response, decode=True)