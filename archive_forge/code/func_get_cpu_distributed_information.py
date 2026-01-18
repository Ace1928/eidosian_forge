import logging
import math
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from shutil import which
from typing import List, Optional
import torch
from packaging.version import parse
def get_cpu_distributed_information() -> CPUInformation:
    """
    Returns various information about the environment in relation to CPU distributed training as a `CPUInformation`
    dataclass.
    """
    information = {}
    information['rank'] = get_int_from_env(['RANK', 'PMI_RANK', 'OMPI_COMM_WORLD_RANK', 'MV2_COMM_WORLD_RANK'], 0)
    information['world_size'] = get_int_from_env(['WORLD_SIZE', 'PMI_SIZE', 'OMPI_COMM_WORLD_SIZE', 'MV2_COMM_WORLD_SIZE'], 1)
    information['local_rank'] = get_int_from_env(['LOCAL_RANK', 'MPI_LOCALRANKID', 'OMPI_COMM_WORLD_LOCAL_RANK', 'MV2_COMM_WORLD_LOCAL_RANK'], 0)
    information['local_world_size'] = get_int_from_env(['LOCAL_WORLD_SIZE', 'MPI_LOCALNRANKS', 'OMPI_COMM_WORLD_LOCAL_SIZE', 'MV2_COMM_WORLD_LOCAL_SIZE'], 1)
    return CPUInformation(**information)