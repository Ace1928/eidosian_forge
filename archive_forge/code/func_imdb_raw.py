import os
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_files
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import mnist
import autokeras as ak
def imdb_raw(num_instances=100):
    dataset = tf.keras.utils.get_file(fname='aclImdb.tar.gz', origin='http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz', extract=True)
    IMDB_DATADIR = os.path.join(os.path.dirname(dataset), 'aclImdb')
    classes = ['pos', 'neg']
    train_data = load_files(os.path.join(IMDB_DATADIR, 'train'), shuffle=True, categories=classes)
    test_data = load_files(os.path.join(IMDB_DATADIR, 'test'), shuffle=False, categories=classes)
    x_train = np.array(train_data.data)
    y_train = np.array(train_data.target)
    x_test = np.array(test_data.data)
    y_test = np.array(test_data.target)
    if num_instances is not None:
        x_train = x_train[:num_instances]
        y_train = y_train[:num_instances]
        x_test = x_test[:num_instances]
        y_test = y_test[:num_instances]
    return ((x_train, y_train), (x_test, y_test))