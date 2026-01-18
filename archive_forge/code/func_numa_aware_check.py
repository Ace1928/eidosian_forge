import glob
import logging
import os
import platform
import re
import subprocess
import sys
from argparse import ArgumentParser, RawTextHelpFormatter, REMAINDER
from os.path import expanduser
from typing import Dict, List
from torch.distributed.elastic.multiprocessing import start_processes, Std
def numa_aware_check(self, core_list):
    """
        Check whether all cores in core_list are in the same NUMA node.

        Cross NUMA will reduce performance.
        We strongly advice to not use cores on different nodes.
        """
    cores_numa_map = self.logical_core_node_map
    numa_ids = []
    for core in core_list:
        numa_id = cores_numa_map[core]
        if numa_id not in numa_ids:
            numa_ids.append(numa_id)
    if len(numa_ids) > 1:
        logger.warning('Numa Aware: cores:%s on different NUMA nodes:%s. To avoid this behavior, please use --ncores-per-instance knob to make sure number of cores is divisible by --ncores-per-instance. Alternatively, please use --skip-cross-node-cores knob.', str(core_list), str(numa_ids))
    if len(numa_ids) == 0:
        raise RuntimeError('invalid number of NUMA nodes; please make sure numa_ids >= 1')
    return numa_ids