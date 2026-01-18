import collections
import enum
from openstack.baremetal.v1 import _common
from openstack import exceptions
from openstack import resource
from openstack import utils
def set_secure_boot(self, session, target):
    """Make a request to change node's secure boot state

        This call is asynchronous, it will return success as soon as the Bare
        Metal service acknowledges the request.

        :param session: The session to use for making this request.
        :param bool target: Boolean indicating secure boot state to set.
            True/False corresponding to 'on'/'off' respectively.
        :returns: ``None``
        :raises: ValueError if ``target`` is not boolean.
        """
    session = self._get_session(session)
    version = utils.pick_microversion(session, _common.CHANGE_BOOT_MODE_VERSION)
    request = self._prepare_request(requires_id=True)
    request.url = utils.urljoin(request.url, 'states', 'secure_boot')
    if not isinstance(target, bool):
        raise ValueError("Invalid target %s. It should be True or False corresponding to secure boot state 'on' or 'off'" % target)
    body = {'target': target}
    response = session.put(request.url, json=body, headers=request.headers, microversion=version, retriable_status_codes=_common.RETRIABLE_STATUS_CODES)
    msg = 'Failed to change secure boot state for {node}'.format(node=self.id)
    exceptions.raise_from_response(response, error_message=msg)