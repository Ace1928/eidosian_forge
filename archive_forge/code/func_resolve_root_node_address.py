import logging
import os
import re
import shutil
import signal
import sys
from typing import Optional
from typing_extensions import override
from lightning_fabric.plugins.environments.cluster_environment import ClusterEnvironment
from lightning_fabric.utilities.imports import _IS_WINDOWS
from lightning_fabric.utilities.rank_zero import rank_zero_warn
from lightning_fabric.utilities.warnings import PossibleUserWarning
@staticmethod
def resolve_root_node_address(nodes: str) -> str:
    """The node selection format in SLURM supports several formats.

        This function selects the first host name from

        - a space-separated list of host names, e.g., 'host0 host1 host3' yields 'host0' as the root
        - a comma-separated list of host names, e.g., 'host0,host1,host3' yields 'host0' as the root
        - the range notation with brackets, e.g., 'host[5-9]' yields 'host5' as the root

        """
    nodes = re.sub('\\[(.*?)[,-].*\\]', '\\1', nodes)
    nodes = re.sub('\\[(.*?)\\]', '\\1', nodes)
    return nodes.split(' ')[0].split(',')[0]