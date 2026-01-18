import dataclasses
import itertools
import os
from typing import Any, Generator, Mapping
from tensorboard.backend.event_processing import event_accumulator
from tensorboard.backend.event_processing import event_file_loader
from tensorboard.backend.event_processing import io_wrapper
from tensorboard.compat import tf
from tensorboard.compat.proto import event_pb2
def get_dict_to_print(field_to_obs):
    """Transform the field-to-obs mapping into a printable dictionary.

    Args:
      field_to_obs: Dict that maps string field to `Observation` list.

    Returns:
      A dict with the keys and values to print to console.
    """

    def compressed_steps(steps):
        return {'num_steps': len(set(steps)), 'min_step': min(steps), 'max_step': max(steps), 'last_step': steps[-1], 'first_step': steps[0], 'outoforder_steps': get_out_of_order(steps)}

    def full_steps(steps):
        return {'steps': steps, 'outoforder_steps': get_out_of_order(steps)}
    output = {}
    for field, observations in field_to_obs.items():
        if not observations:
            output[field] = None
            continue
        steps = [x['step'] for x in observations]
        if field in SHORT_FIELDS:
            output[field] = compressed_steps(steps)
        if field in LONG_FIELDS:
            output[field] = full_steps(steps)
    return output