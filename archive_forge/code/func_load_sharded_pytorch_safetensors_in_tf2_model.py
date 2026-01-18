import os
import re
import numpy
from .utils import (
from .utils import transpose as transpose_func
def load_sharded_pytorch_safetensors_in_tf2_model(tf_model, safetensors_shards, tf_inputs=None, allow_missing_keys=False, output_loading_info=False, _prefix=None, tf_to_pt_weight_rename=None, ignore_mismatched_sizes=False):
    all_loading_infos = []
    for shard in safetensors_shards:
        with safe_open(shard, framework='tf') as safetensors_archive:
            tf_model, loading_info = load_pytorch_state_dict_in_tf2_model(tf_model, safetensors_archive, tf_inputs=tf_inputs, allow_missing_keys=allow_missing_keys, output_loading_info=True, _prefix=_prefix, tf_to_pt_weight_rename=tf_to_pt_weight_rename, ignore_mismatched_sizes=ignore_mismatched_sizes, skip_logger_warnings=True)
        all_loading_infos.append(loading_info)
    missing_keys = sorted(set.intersection(*[set(info['missing_keys']) for info in all_loading_infos]))
    unexpected_keys = sum([info['unexpected_keys'] for info in all_loading_infos], [])
    mismatched_keys = sum([info['mismatched_keys'] for info in all_loading_infos], [])
    _log_key_warnings(missing_keys, unexpected_keys, mismatched_keys, class_name=tf_model.__class__.__name__)
    if output_loading_info:
        loading_info = {'missing_keys': missing_keys, 'unexpected_keys': unexpected_keys, 'mismatched_keys': mismatched_keys}
        return (tf_model, loading_info)
    return tf_model