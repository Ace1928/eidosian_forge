import tensorflow.compat.v2 as tf
from absl import flags
def get_tpu_cluster_resolver():
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=FLAGS.tpu, zone=FLAGS.zone, project=FLAGS.project)
    return resolver