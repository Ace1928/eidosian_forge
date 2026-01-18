from keras.src.applications import imagenet_utils
from keras.src.applications import resnet
from tensorflow.python.util.tf_export import keras_export
@keras_export('keras.applications.resnet_v2.preprocess_input')
def preprocess_input(x, data_format=None):
    return imagenet_utils.preprocess_input(x, data_format=data_format, mode='tf')