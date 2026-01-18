from __future__ import annotations
import copy
from typing import Any, Callable, Dict, List, Union
import torch
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.qconfig_mapping import _QCONFIG_STYLE_ORDER
from torch.ao.quantization.qconfig import QConfigAny
@classmethod
def from_list_qconfig_mapping(cls, qconfig_mapping_list: List[QConfigMapping]) -> QConfigMultiMapping:
    """
        Creates a QConfigMultiMapping from a list of QConfigMappings
        """
    new_qconfig_multi_mapping = cls()
    new_qconfig_multi_mapping.qconfig_mappings_list = copy.deepcopy(qconfig_mapping_list)
    for style in _QCONFIG_STYLE_ORDER[1:]:
        qconfig_dict_list: Dict[Any, List[QConfigAny]] = {}
        for qconfig_mapping in qconfig_mapping_list:
            qconfig_dict = getattr(qconfig_mapping, style)
            for key, qconfig in qconfig_dict.items():
                if key not in qconfig_dict_list:
                    qconfig_dict_list[key] = []
                qconfig_dict_list[key].append(qconfig)
        set_method_name = _QCONFIG_STYLE_TO_METHOD[style]
        set_method = getattr(new_qconfig_multi_mapping, set_method_name)
        for key, qconfig_list in qconfig_dict_list.items():
            if isinstance(key, tuple):
                set_method(*key, qconfig_list)
            else:
                set_method(key, qconfig_list)
    return new_qconfig_multi_mapping