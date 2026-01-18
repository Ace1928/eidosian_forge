import struct
from oslo_log import log as logging
def get_inspector(format_name):
    """Returns a FormatInspector class based on the given name.

    :param format_name: The name of the disk_format (raw, qcow2, etc).
    :returns: A FormatInspector or None if unsupported.
    """
    formats = {'raw': FileInspector, 'qcow2': QcowInspector, 'vhd': VHDInspector, 'vhdx': VHDXInspector, 'vmdk': VMDKInspector, 'vdi': VDIInspector}
    return formats.get(format_name)