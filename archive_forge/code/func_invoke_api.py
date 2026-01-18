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
def invoke_api(self, module, method, *args, **kwargs):
    """Wrapper method for invoking APIs.

        The API call is retried in the event of exceptions due to session
        overload or connection problems.

        :param module: module corresponding to the VIM API call
        :param method: method in the module which corresponds to the
                       VIM API call
        :param args: arguments to the method
        :param kwargs: keyword arguments to the method
        :returns: response from the API call
        :raises: VimException, VimFaultException, VimAttributeException,
                 VimSessionOverLoadException, VimConnectionException
        """

    @RetryDecorator(max_retry_count=self._api_retry_count, exceptions=(exceptions.VimSessionOverLoadException, exceptions.VimConnectionException))
    def _invoke_api(module, method, *args, **kwargs):
        try:
            api_method = getattr(module, method)
            return api_method(*args, **kwargs)
        except exceptions.VimFaultException as excep:
            if exceptions.NOT_AUTHENTICATED in excep.fault_list:
                if self.is_current_session_active():
                    LOG.debug('Returning empty response for %(module)s.%(method)s invocation.', {'module': module, 'method': method})
                    return []
                else:
                    excep_msg = _('Current session: %(session)s is inactive; re-creating the session while invoking method %(module)s.%(method)s.') % {'session': _trunc_id(self._session_id), 'module': module, 'method': method}
                    LOG.debug(excep_msg)
                    self._create_session()
                    raise exceptions.VimConnectionException(excep_msg, excep)
            else:
                if excep.fault_list:
                    LOG.debug('Fault list: %s', excep.fault_list)
                    fault = excep.fault_list[0]
                    clazz = exceptions.get_fault_class(fault)
                    if clazz:
                        raise clazz(str(excep), details=excep.details)
                raise
        except exceptions.VimConnectionException:
            with excutils.save_and_reraise_exception():
                if not self.is_current_session_active():
                    LOG.debug('Re-creating session due to connection problems while invoking method %(module)s.%(method)s.', {'module': module, 'method': method})
                    self._create_session()
    return _invoke_api(module, method, *args, **kwargs)