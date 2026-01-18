from dataclasses import dataclass, field
import pathlib
import pprint
from typing import (
from ray.util.annotations import PublicAPI
from ray.rllib.utils.annotations import override, ExperimentalAPI
from ray.rllib.utils.nested_dict import NestedDict
from ray.rllib.core.models.specs.typing import SpecType
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.core.rl_module.rl_module import (
from ray.rllib.utils.annotations import OverrideToImplementCustomLogic
from ray.rllib.utils.policy import validate_policy_id
from ray.rllib.utils.serialization import serialize_type, deserialize_type
from ray.rllib.utils.typing import T
def add_modules(self, module_specs: Dict[ModuleID, SingleAgentRLModuleSpec], overwrite: bool=True) -> None:
    """Add new module specs to the spec or updates existing ones.

        Args:
            module_specs: The mapping for the module_id to the single-agent module
                specs to be added to this multi-agent module spec.
            overwrite: Whether to overwrite the existing module specs if they already
                exist. If False, they will be updated only.
        """
    if self.module_specs is None:
        self.module_specs = {}
    for module_id, module_spec in module_specs.items():
        if overwrite or module_id not in self.module_specs:
            self.module_specs[module_id] = module_spec
        else:
            self.module_specs[module_id].update(module_spec)