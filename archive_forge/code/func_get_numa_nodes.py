import socket
from oslo_log import log as logging
from os_win._i18n import _
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
from os_win.utils.winapi import libs as w_lib
def get_numa_nodes(self):
    """Returns the host's list of NUMA nodes.

        :returns: list of dictionaries containing information about each
            host NUMA node. Each host has at least one NUMA node.
        """
    numa_nodes = self._conn.Msvm_NumaNode()
    nodes_info = []
    system_memory = self._conn.Msvm_Memory(['NumberOfBlocks'])
    processors = self._conn.Msvm_Processor(['DeviceID'])
    for node in numa_nodes:
        numa_assoc = self._conn.Msvm_HostedDependency(Antecedent=node.path_())
        numa_node_assoc = [item.Dependent for item in numa_assoc]
        memory_info = self._get_numa_memory_info(numa_node_assoc, system_memory)
        if not memory_info:
            LOG.warning('Could not find memory information for NUMA node. Skipping node measurements.')
            continue
        cpu_info = self._get_numa_cpu_info(numa_node_assoc, processors)
        if not cpu_info:
            LOG.warning('Could not find CPU information for NUMA node. Skipping node measurements.')
            continue
        node_info = {'id': node.NodeID.split('\\')[-1], 'memory': memory_info.NumberOfBlocks, 'memory_usage': node.CurrentlyConsumableMemoryBlocks, 'cpuset': set([c.DeviceID.split('\\')[-1] for c in cpu_info]), 'cpu_usage': 0}
        nodes_info.append(node_info)
    return nodes_info