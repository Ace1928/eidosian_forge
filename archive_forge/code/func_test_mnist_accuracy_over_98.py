import os
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_files
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import mnist
import autokeras as ak
def test_mnist_accuracy_over_98(tmp_path):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    clf = ak.ImageClassifier(max_trials=1, directory=tmp_path)
    clf.fit(x_train, y_train, epochs=10)
    accuracy = clf.evaluate(x_test, y_test)[1]
    assert accuracy >= 0.98