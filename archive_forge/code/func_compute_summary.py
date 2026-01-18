import numpy as np
from tensorboard.plugins.pr_curve import metadata
def compute_summary(tp, fp, tn, fn, collections):
    precision = tp / tf.maximum(_MINIMUM_COUNT, tp + fp)
    recall = tp / tf.maximum(_MINIMUM_COUNT, tp + fn)
    return _create_tensor_summary(name, tp, fp, tn, fn, precision, recall, num_thresholds, display_name, description, collections)