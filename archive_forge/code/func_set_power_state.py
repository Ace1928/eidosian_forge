import collections
import enum
from openstack.baremetal.v1 import _common
from openstack import exceptions
from openstack import resource
from openstack import utils
def set_power_state(self, session, target, wait=False, timeout=None):
    """Run an action modifying this node's power state.

        This call is asynchronous, it will return success as soon as the Bare
        Metal service acknowledges the request.

        :param session: The session to use for making this request.
        :type session: :class:`~keystoneauth1.adapter.Adapter`
        :param target: Target power state, as a :class:`PowerAction` or
            a string.
        :param wait: Whether to wait for the expected power state to be
            reached.
        :param timeout: Timeout (in seconds) to wait for the target state to be
            reached. If ``None``, wait without timeout.
        """
    if isinstance(target, PowerAction):
        target = target.value
    if wait:
        try:
            expected = _common.EXPECTED_POWER_STATES[target]
        except KeyError:
            raise ValueError('Cannot use target power state %s with wait, the expected state is not known' % target)
    session = self._get_session(session)
    if target.startswith('soft '):
        version = '1.27'
    else:
        version = None
    version = self._assert_microversion_for(session, 'commit', version)
    body = {'target': target}
    request = self._prepare_request(requires_id=True)
    request.url = utils.urljoin(request.url, 'states', 'power')
    response = session.put(request.url, json=body, headers=request.headers, microversion=version, retriable_status_codes=_common.RETRIABLE_STATUS_CODES)
    msg = 'Failed to set power state for bare metal node {node} to {target}'.format(node=self.id, target=target)
    exceptions.raise_from_response(response, error_message=msg)
    if wait:
        self.wait_for_power_state(session, expected, timeout=timeout)