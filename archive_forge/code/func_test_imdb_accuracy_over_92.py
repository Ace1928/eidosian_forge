import os
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_files
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import mnist
import autokeras as ak
def test_imdb_accuracy_over_92(tmp_path):
    (x_train, y_train), (x_test, y_test) = imdb_raw(num_instances=None)
    clf = ak.TextClassifier(max_trials=3, directory=tmp_path)
    clf.fit(x_train, y_train, batch_size=6, epochs=1)
    accuracy = clf.evaluate(x_test, y_test)[1]
    assert accuracy >= 0.92