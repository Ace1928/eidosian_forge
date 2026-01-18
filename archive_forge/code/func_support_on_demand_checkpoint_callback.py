import os
import requests
import tensorflow.compat.v2 as tf
def support_on_demand_checkpoint_callback(strategy):
    if _on_gcp() and isinstance(strategy, tf.distribute.MultiWorkerMirroredStrategy):
        return True
    return False