import json
from typing import Any, Dict, NewType, Optional, Sequence
from wandb.proto import wandb_internal_pb2
from wandb.sdk.lib import proto_util, telemetry
def merge_resumed_config(self, old_config_tree: Dict[str, Any]) -> None:
    """Merges the config from a run that's being resumed."""
    self._add_unset_keys_from_subtree(old_config_tree, [])
    self._add_unset_keys_from_subtree(old_config_tree, [_WANDB_INTERNAL_KEY, 'visualize'])
    self._add_unset_keys_from_subtree(old_config_tree, [_WANDB_INTERNAL_KEY, 'viz'])