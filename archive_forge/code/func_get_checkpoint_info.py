import logging
import json
import os
from packaging import version
import re
from typing import Any, Dict, Union
import ray
from ray.rllib.utils.serialization import NOT_SERIALIZABLE, serialize_type
from ray.train import Checkpoint
from ray.util import log_once
from ray.util.annotations import PublicAPI
@PublicAPI(stability='alpha')
def get_checkpoint_info(checkpoint: Union[str, Checkpoint]) -> Dict[str, Any]:
    """Returns a dict with information about a Algorithm/Policy checkpoint.

    If the given checkpoint is a >=v1.0 checkpoint directory, try reading all
    information from the contained `rllib_checkpoint.json` file.

    Args:
        checkpoint: The checkpoint directory (str) or an AIR Checkpoint object.

    Returns:
        A dict containing the keys:
        "type": One of "Policy" or "Algorithm".
        "checkpoint_version": A version tuple, e.g. v1.0, indicating the checkpoint
        version. This will help RLlib to remain backward compatible wrt. future
        Ray and checkpoint versions.
        "checkpoint_dir": The directory with all the checkpoint files in it. This might
        be the same as the incoming `checkpoint` arg.
        "state_file": The main file with the Algorithm/Policy's state information in it.
        This is usually a pickle-encoded file.
        "policy_ids": An optional set of PolicyIDs in case we are dealing with an
        Algorithm checkpoint. None if `checkpoint` is a Policy checkpoint.
    """
    info = {'type': 'Algorithm', 'format': 'cloudpickle', 'checkpoint_version': CHECKPOINT_VERSION, 'checkpoint_dir': None, 'state_file': None, 'policy_ids': None}
    if isinstance(checkpoint, Checkpoint):
        checkpoint: str = checkpoint.to_directory()
    if os.path.isdir(checkpoint):
        info.update({'checkpoint_dir': checkpoint})
        for file in os.listdir(checkpoint):
            path_file = os.path.join(checkpoint, file)
            if os.path.isfile(path_file):
                if re.match('checkpoint-\\d+', file):
                    info.update({'checkpoint_version': version.Version('0.1'), 'state_file': path_file})
                    return info
        if os.path.isfile(os.path.join(checkpoint, 'rllib_checkpoint.json')):
            with open(os.path.join(checkpoint, 'rllib_checkpoint.json')) as f:
                rllib_checkpoint_info = json.load(fp=f)
            if 'checkpoint_version' in rllib_checkpoint_info:
                rllib_checkpoint_info['checkpoint_version'] = version.Version(rllib_checkpoint_info['checkpoint_version'])
            info.update(rllib_checkpoint_info)
        elif log_once('no_rllib_checkpoint_json_file'):
            logger.warning(f'No `rllib_checkpoint.json` file found in checkpoint directory {checkpoint}! Trying to extract checkpoint info from other files found in that dir.')
        for extension in ['pkl', 'msgpck']:
            if os.path.isfile(os.path.join(checkpoint, 'policy_state.' + extension)):
                info.update({'type': 'Policy', 'format': 'cloudpickle' if extension == 'pkl' else 'msgpack', 'checkpoint_version': CHECKPOINT_VERSION, 'state_file': os.path.join(checkpoint, f'policy_state.{extension}')})
                return info
        format = None
        for extension in ['pkl', 'msgpck']:
            state_file = os.path.join(checkpoint, f'algorithm_state.{extension}')
            if os.path.isfile(state_file):
                format = 'cloudpickle' if extension == 'pkl' else 'msgpack'
                break
        if format is None:
            raise ValueError('Given checkpoint does not seem to be valid! No file with the name `algorithm_state.[pkl|msgpck]` (or `checkpoint-[0-9]+`) found.')
        info.update({'format': format, 'state_file': state_file})
        policies_dir = os.path.join(checkpoint, 'policies')
        if os.path.isdir(policies_dir):
            policy_ids = set()
            for policy_id in os.listdir(policies_dir):
                policy_ids.add(policy_id)
            info.update({'policy_ids': policy_ids})
    elif os.path.isfile(checkpoint):
        info.update({'checkpoint_version': version.Version('0.1'), 'checkpoint_dir': os.path.dirname(checkpoint), 'state_file': checkpoint})
    else:
        raise ValueError(f'Given checkpoint ({checkpoint}) not found! Must be a checkpoint directory (or a file for older checkpoint versions).')
    return info