import logging
from oslo_vmware._i18n import _
def register_fault_class(name, exception):
    fault_class = _fault_classes_registry.get(name)
    if not issubclass(exception, VimException):
        raise TypeError(_('exception should be a subclass of VimException'))
    if fault_class:
        LOG.debug('Overriding exception for %s', name)
    _fault_classes_registry[name] = exception