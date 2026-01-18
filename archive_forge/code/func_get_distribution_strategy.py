import json
import os
import tensorflow.compat.v2 as tf
def get_distribution_strategy(distribution_strategy='mirrored', num_gpus=0, all_reduce_alg=None, num_packs=1):
    """Return a DistributionStrategy for running the model.

    Args:
      distribution_strategy: a string specifying which distribution strategy to
        use. Accepted values are "off", "one_device", "mirrored", and
        "multi_worker_mirrored" -- case insensitive. "off" means not to use
        Distribution Strategy.
      num_gpus: Number of GPUs to run this model.

    Returns:
      tf.distribute.DistibutionStrategy object.
    Raises:
      ValueError: if `distribution_strategy` is "off" or "one_device" and
        `num_gpus` is larger than 1; or `num_gpus` is negative.
    """
    if num_gpus < 0:
        raise ValueError('`num_gpus` can not be negative.')
    distribution_strategy = distribution_strategy.lower()
    if distribution_strategy == 'off':
        if num_gpus > 1:
            raise ValueError('When {} GPUs are specified, distribution_strategy flag cannot be set to `off`.'.format(num_gpus))
        return None
    if distribution_strategy == 'multi_worker_mirrored':
        return tf.distribute.experimental.MultiWorkerMirroredStrategy(communication=_collective_communication(all_reduce_alg))
    if distribution_strategy == 'one_device':
        if num_gpus == 0:
            return tf.distribute.OneDeviceStrategy('device:CPU:0')
        if num_gpus > 1:
            raise ValueError('`OneDeviceStrategy` can not be used for more than one device.')
        return tf.distribute.OneDeviceStrategy('device:GPU:0')
    if distribution_strategy == 'mirrored':
        if num_gpus == 0:
            devices = ['device:CPU:0']
        else:
            devices = ['device:GPU:%d' % i for i in range(num_gpus)]
        return tf.distribute.MirroredStrategy(devices=devices, cross_device_ops=_mirrored_cross_device_ops(all_reduce_alg, num_packs))
    raise ValueError(f'Unrecognized Distribution Strategy: {distribution_strategy}')