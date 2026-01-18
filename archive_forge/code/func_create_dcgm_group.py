from typing import List, Optional, Set, Tuple, Union
import dcgm_fields
import torch
from dcgm_fields import DcgmFieldGetById
from dcgm_structs import DCGM_GROUP_EMPTY, DCGM_OPERATION_MODE_AUTO
from pydcgm import DcgmFieldGroup, DcgmGroup, DcgmHandle
from .profiler import _Profiler, logger
def create_dcgm_group(self, gpus_to_profile: Union[Tuple[int], Tuple[int, ...]]) -> Optional[DcgmGroup]:
    if self.dcgmHandle is None:
        return None
    dcgmSystem = self.dcgmHandle.GetSystem()
    supportedGPUs = dcgmSystem.discovery.GetAllSupportedGpuIds()
    valid_gpus_to_profile: List[int] = [gpu for gpu in gpus_to_profile if gpu in supportedGPUs]
    if len(valid_gpus_to_profile) < 1:
        logger.warning(f'The provided GPUs are not supported on this system: provided {gpus_to_profile}, supported {supportedGPUs}. No data will be captured.')
        return None
    dcgmGroup = DcgmGroup(self.dcgmHandle, groupName='DCGMProfiler', groupType=DCGM_GROUP_EMPTY)
    for gpu in valid_gpus_to_profile:
        dcgmGroup.AddGpu(gpu)
    return dcgmGroup