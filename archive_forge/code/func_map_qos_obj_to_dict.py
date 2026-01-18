from __future__ import (absolute_import, division, print_function)
from ansible.module_utils._text import to_native
@staticmethod
def map_qos_obj_to_dict(qos_obj):
    """ Take a QOS object and return a key, normalize the key names
            Interestingly, the APIs are using different ids for create and get
        """
    mappings = [('burst_iops', 'burstIOPS'), ('min_iops', 'minIOPS'), ('max_iops', 'maxIOPS')]
    qos_dict = vars(qos_obj)
    for read, send in mappings:
        if read in qos_dict:
            qos_dict[send] = qos_dict.pop(read)
    return qos_dict