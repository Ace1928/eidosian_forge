import json
import numpy as np
from keras.src.preprocessing.sequence import _remove_long_seq
from keras.src.utils.data_utils import get_file
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export
@keras_export('keras.datasets.reuters.get_label_names')
def get_label_names():
    """Returns labels as a list of strings with indices matching training data.

    Reference:

    - [Reuters Dataset](https://martin-thoma.com/nlp-reuters/)
    """
    return ('cocoa', 'grain', 'veg-oil', 'earn', 'acq', 'wheat', 'copper', 'housing', 'money-supply', 'coffee', 'sugar', 'trade', 'reserves', 'ship', 'cotton', 'carcass', 'crude', 'nat-gas', 'cpi', 'money-fx', 'interest', 'gnp', 'meal-feed', 'alum', 'oilseed', 'gold', 'tin', 'strategic-metal', 'livestock', 'retail', 'ipi', 'iron-steel', 'rubber', 'heat', 'jobs', 'lei', 'bop', 'zinc', 'orange', 'pet-chem', 'dlr', 'gas', 'silver', 'wpi', 'hog', 'lead')