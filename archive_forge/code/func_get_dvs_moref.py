import logging
from oslo_vmware import vim_util
def get_dvs_moref(value):
    """Get managed DVS object reference.

    :param value: value of the DVS managed object
    :returns: managed object reference with given value and type
              'VmwareDistributedVirtualSwitch'
    """
    return vim_util.get_moref(value, 'VmwareDistributedVirtualSwitch')