from .. import auth, errors, utils
from ..types import ServiceMode
@utils.minimum_version('1.24')
def create_service(self, task_template, name=None, labels=None, mode=None, update_config=None, networks=None, endpoint_config=None, endpoint_spec=None, rollback_config=None):
    """
        Create a service.

        Args:
            task_template (TaskTemplate): Specification of the task to start as
                part of the new service.
            name (string): User-defined name for the service. Optional.
            labels (dict): A map of labels to associate with the service.
                Optional.
            mode (ServiceMode): Scheduling mode for the service (replicated
                or global). Defaults to replicated.
            update_config (UpdateConfig): Specification for the update strategy
                of the service. Default: ``None``
            rollback_config (RollbackConfig): Specification for the rollback
                strategy of the service. Default: ``None``
            networks (:py:class:`list`): List of network names or IDs or
                :py:class:`~docker.types.NetworkAttachmentConfig` to attach the
                service to. Default: ``None``.
            endpoint_spec (EndpointSpec): Properties that can be configured to
                access and load balance a service. Default: ``None``.

        Returns:
            A dictionary containing an ``ID`` key for the newly created
            service.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.
        """
    _check_api_features(self._version, task_template, update_config, endpoint_spec, rollback_config)
    url = self._url('/services/create')
    headers = {}
    image = task_template.get('ContainerSpec', {}).get('Image', None)
    if image is None:
        raise errors.DockerException('Missing mandatory Image key in ContainerSpec')
    if mode and (not isinstance(mode, dict)):
        mode = ServiceMode(mode)
    registry, repo_name = auth.resolve_repository_name(image)
    auth_header = auth.get_config_header(self, registry)
    if auth_header:
        headers['X-Registry-Auth'] = auth_header
    if utils.version_lt(self._version, '1.25'):
        networks = networks or task_template.pop('Networks', None)
    data = {'Name': name, 'Labels': labels, 'TaskTemplate': task_template, 'Mode': mode, 'Networks': utils.convert_service_networks(networks), 'EndpointSpec': endpoint_spec}
    if update_config is not None:
        data['UpdateConfig'] = update_config
    if rollback_config is not None:
        data['RollbackConfig'] = rollback_config
    return self._result(self._post_json(url, data=data, headers=headers), True)