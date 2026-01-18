from warnings import simplefilter
import numpy as np
from sklearn.utils.multiclass import unique_labels
import wandb
from wandb.sklearn import utils
def get_named_labels(labels, numeric_labels):
    return np.array([labels[num_label] for num_label in numeric_labels])