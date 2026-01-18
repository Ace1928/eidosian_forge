import logging
from oslo_concurrency import lockutils
from oslo_context import context
from oslo_utils import excutils
from oslo_utils import reflection
from oslo_vmware._i18n import _
from oslo_vmware.common import loopingcall
from oslo_vmware import exceptions
from oslo_vmware import pbm
from oslo_vmware import vim
from oslo_vmware import vim_util
def is_current_session_active(self):
    """Check if current session is active.

        :returns: True if the session is active; False otherwise
        """
    LOG.debug('Checking if the current session: %s is active.', _trunc_id(self._session_id))
    is_active = False
    try:
        is_active = self.vim.SessionIsActive(self.vim.service_content.sessionManager, sessionID=self._session_id, userName=self._session_username)
    except exceptions.VimException as ex:
        LOG.debug('Error: %(error)s occurred while checking whether the current session: %(session)s is active.', {'error': str(ex), 'session': _trunc_id(self._session_id)})
    return is_active