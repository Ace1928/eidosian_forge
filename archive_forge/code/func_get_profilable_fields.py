from typing import List, Optional, Set, Tuple, Union
import dcgm_fields
import torch
from dcgm_fields import DcgmFieldGetById
from dcgm_structs import DCGM_GROUP_EMPTY, DCGM_OPERATION_MODE_AUTO
from pydcgm import DcgmFieldGroup, DcgmGroup, DcgmHandle
from .profiler import _Profiler, logger
def get_profilable_fields(self) -> Set[int]:
    assert self.dcgmGroup is not None
    dcgmMetricGroups = self.dcgmGroup.profiling.GetSupportedMetricGroups()
    profilableFieldIds = set()
    for group_idx in range(dcgmMetricGroups.numMetricGroups):
        metric_group = dcgmMetricGroups.metricGroups[group_idx]
        for field_id in metric_group.fieldIds[:metric_group.numFieldIds]:
            profilableFieldIds.add(field_id)
    return profilableFieldIds