from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import numpy as np
from six import string_types
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.inputs.queues import feeding_functions
def input_fn():
    """Numpy input function."""
    ordered_dict_data = _validate_and_convert_features(x)
    feature_keys = list(ordered_dict_data.keys())
    if y is None:
        target_keys = None
    elif isinstance(y, dict):
        if not y:
            raise ValueError('y cannot be empty dict, use None instead.')
        ordered_dict_y = collections.OrderedDict(sorted(y.items(), key=lambda t: t[0]))
        target_keys = list(ordered_dict_y.keys())
        duplicate_keys = set(feature_keys).intersection(set(target_keys))
        if duplicate_keys:
            raise ValueError('{} duplicate keys are found in both x and y: {}'.format(len(duplicate_keys), duplicate_keys))
        ordered_dict_data.update(ordered_dict_y)
    else:
        target_keys = _get_unique_target_key(ordered_dict_data)
        ordered_dict_data[target_keys] = y
    if len(set((v.shape[0] for v in ordered_dict_data.values()))) != 1:
        shape_dict_of_x = {k: ordered_dict_data[k].shape for k in feature_keys}
        if target_keys is None:
            shape_of_y = None
        elif isinstance(target_keys, string_types):
            shape_of_y = y.shape
        else:
            shape_of_y = {k: ordered_dict_data[k].shape for k in target_keys}
        raise ValueError('Length of tensors in x and y is mismatched. All elements in x and y must have the same length.\nShapes in x: {}\nShapes in y: {}\n'.format(shape_dict_of_x, shape_of_y))
    queue = feeding_functions._enqueue_data(ordered_dict_data, queue_capacity, shuffle=shuffle, num_threads=num_threads, enqueue_size=batch_size, num_epochs=num_epochs)
    batch = queue.dequeue_many(batch_size) if num_epochs is None else queue.dequeue_up_to(batch_size)
    if batch:
        batch.pop(0)
    if isinstance(x, np.ndarray):
        features = batch[0]
    else:
        features = dict(zip(feature_keys, batch[:len(feature_keys)]))
    if target_keys is None:
        return features
    elif isinstance(target_keys, string_types):
        target = batch[-1]
        return (features, target)
    else:
        target = dict(zip(target_keys, batch[-len(target_keys):]))
        return (features, target)