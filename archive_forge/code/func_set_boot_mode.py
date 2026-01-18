import collections
import enum
from openstack.baremetal.v1 import _common
from openstack import exceptions
from openstack import resource
from openstack import utils
def set_boot_mode(self, session, target):
    """Make a request to change node's boot mode

        This call is asynchronous, it will return success as soon as the Bare
        Metal service acknowledges the request.

        :param session: The session to use for making this request.
        :param target: Boot mode to set for node, one of either 'uefi'/'bios'.
        :returns: ``None``
        :raises: ValueError if ``target`` is not one of 'uefi or 'bios'.
        """
    session = self._get_session(session)
    version = utils.pick_microversion(session, _common.CHANGE_BOOT_MODE_VERSION)
    request = self._prepare_request(requires_id=True)
    request.url = utils.urljoin(request.url, 'states', 'boot_mode')
    if target not in ('uefi', 'bios'):
        raise ValueError("Unrecognized boot mode %s.Boot mode should be one of 'uefi' or 'bios'." % target)
    body = {'target': target}
    response = session.put(request.url, json=body, headers=request.headers, microversion=version, retriable_status_codes=_common.RETRIABLE_STATUS_CODES)
    msg = 'Failed to change boot mode for node {node}'.format(node=self.id)
    exceptions.raise_from_response(response, error_message=msg)