import logging
from oslo_vmware._i18n import _
def get_fault_class(name):
    """Get a named subclass of VimException."""
    name = str(name)
    fault_class = _fault_classes_registry.get(name)
    if not fault_class:
        LOG.debug('Fault %s not matched.', name)
    return fault_class