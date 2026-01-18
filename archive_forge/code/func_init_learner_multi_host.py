import collections
import pickle
import threading
import time
import timeit
from absl import flags
from absl import logging
import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.distribute import values as values_lib  
from tensorflow.python.framework import composite_tensor  
from tensorflow.python.framework import tensor_conversion_registry  
def init_learner_multi_host(num_training_tpus: int):
    """Performs common learner initialization including multi-host setting.

  In multi-host setting, this function will enter a loop for secondary learners
  until the primary learner signals end of training.

  Args:
    num_training_tpus: Number of training TPUs.

  Returns:
    A MultiHostSettings object.
  """
    tpu = ''
    job_name = None
    if tf.config.experimental.list_logical_devices('TPU'):
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu, job_name=job_name)
        topology = tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.experimental.TPUStrategy(resolver)
        assert num_training_tpus % topology.num_tasks == 0
        num_training_tpus_per_task = num_training_tpus // topology.num_tasks
        hosts = []
        training_coordinates = []
        for per_host_coordinates in topology.device_coordinates:
            host = topology.cpu_device_name_at_coordinates(per_host_coordinates[0], job=job_name)
            task_training_coordinates = per_host_coordinates[:num_training_tpus_per_task]
            training_coordinates.extend([[c] for c in task_training_coordinates])
            inference_coordinates = per_host_coordinates[num_training_tpus_per_task:]
            hosts.append((host, [topology.tpu_device_name_at_coordinates(c, job=job_name) for c in inference_coordinates]))
        training_da = tf.tpu.experimental.DeviceAssignment(topology, training_coordinates)
        training_strategy = tf.distribute.experimental.TPUStrategy(resolver, device_assignment=training_da)
        return MultiHostSettings(strategy, hosts, training_strategy, tpu_encode, tpu_decode)
    else:
        tf.device('/cpu').__enter__()
        any_gpu = tf.config.experimental.list_logical_devices('GPU')
        device_name = '/device:GPU:0' if any_gpu else '/device:CPU:0'
        strategy = tf.distribute.OneDeviceStrategy(device=device_name)
        enc = lambda x: x
        dec = lambda x, s=None: x if s is None else tf.nest.pack_sequence_as(s, x)
        return MultiHostSettings(strategy, [('/cpu', [device_name])], strategy, enc, dec)