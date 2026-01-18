from .. import auth, errors, utils
from ..types import ServiceMode
@utils.minimum_version('1.25')
@utils.check_resource('service')
def service_logs(self, service, details=False, follow=False, stdout=False, stderr=False, since=0, timestamps=False, tail='all', is_tty=None):
    """
            Get log stream for a service.
            Note: This endpoint works only for services with the ``json-file``
            or ``journald`` logging drivers.

            Args:
                service (str): ID or name of the service
                details (bool): Show extra details provided to logs.
                    Default: ``False``
                follow (bool): Keep connection open to read logs as they are
                    sent by the Engine. Default: ``False``
                stdout (bool): Return logs from ``stdout``. Default: ``False``
                stderr (bool): Return logs from ``stderr``. Default: ``False``
                since (int): UNIX timestamp for the logs staring point.
                    Default: 0
                timestamps (bool): Add timestamps to every log line.
                tail (string or int): Number of log lines to be returned,
                    counting from the current end of the logs. Specify an
                    integer or ``'all'`` to output all log lines.
                    Default: ``all``
                is_tty (bool): Whether the service's :py:class:`ContainerSpec`
                    enables the TTY option. If omitted, the method will query
                    the Engine for the information, causing an additional
                    roundtrip.

            Returns (generator): Logs for the service.
        """
    params = {'details': details, 'follow': follow, 'stdout': stdout, 'stderr': stderr, 'since': since, 'timestamps': timestamps, 'tail': tail}
    url = self._url('/services/{0}/logs', service)
    res = self._get(url, params=params, stream=True)
    if is_tty is None:
        is_tty = self.inspect_service(service)['Spec']['TaskTemplate']['ContainerSpec'].get('TTY', False)
    return self._get_result_tty(True, res, is_tty)