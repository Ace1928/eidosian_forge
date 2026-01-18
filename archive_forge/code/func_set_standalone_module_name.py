from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.backend_config import BackendConfig
from torch.ao.quantization.quant_type import QuantType, _quant_type_from_str, _get_quant_type_to_str
def set_standalone_module_name(self, module_name: str, qconfig_mapping: Optional[QConfigMapping], example_inputs: Tuple[Any, ...], prepare_custom_config: Optional[PrepareCustomConfig], backend_config: Optional[BackendConfig]) -> PrepareCustomConfig:
    """
        Set the configuration for running a standalone module identified by ``module_name``.

        If ``qconfig_mapping`` is None, the parent ``qconfig_mapping`` will be used instead.
        If ``prepare_custom_config`` is None, an empty ``PrepareCustomConfig`` will be used.
        If ``backend_config`` is None, the parent ``backend_config`` will be used instead.
        """
    self.standalone_module_names[module_name] = StandaloneModuleConfigEntry(qconfig_mapping, example_inputs, prepare_custom_config, backend_config)
    return self