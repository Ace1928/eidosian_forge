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
def step_end(self, session, strategy=None, step_increment=1):
    logs = []
    for value in session:
        if strategy:
            value = tf.reduce_mean(tf.cast(strategy.experimental_local_results(value)[0], tf.float32))
        logs.append(value)
    self.ready_values.assign(logs)
    self.step_cnt.assign_add(step_increment)