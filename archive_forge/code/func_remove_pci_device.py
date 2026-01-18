import re
from oslo_log import log as logging
import six
from os_win._i18n import _
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
from oslo_utils import units
def remove_pci_device(self, vm_name, vendor_id, product_id):
    """Removes the given PCI device from the given VM.

        :param vm_name: the name of the VM from which the PCI device will be
            attached from.
        :param vendor_id: the PCI device's vendor ID.
        :param product_id: the PCI device's product ID.
        """
    vmsettings = self._lookup_vm_check(vm_name)
    pattern = re.compile('^(.*)VEN_%(vendor_id)s&DEV_%(product_id)s&(.*)$' % {'vendor_id': vendor_id, 'product_id': product_id}, re.IGNORECASE)
    pci_sds = _wqlutils.get_element_associated_class(self._conn, self._PCI_EXPRESS_SETTING_DATA, vmsettings.InstanceID)
    pci_sds = [sd for sd in pci_sds if pattern.match(sd.HostResource[0])]
    if pci_sds:
        self._jobutils.remove_virt_resource(pci_sds[0])
    else:
        LOG.debug('PCI device with vendor ID %(vendor_id)s and %(product_id)s is not attached to %(vm_name)s', {'vendor_id': vendor_id, 'product_id': product_id, 'vm_name': vm_name})