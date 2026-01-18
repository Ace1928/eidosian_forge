import argparse
import json
import os
import numpy as np
import tensorflow as tf
from filelock import FileLock
from ray.air.integrations.keras import ReportCheckpointCallback
from ray.train import Result, RunConfig, ScalingConfig
from ray.train.tensorflow import TensorflowTrainer
def build_cnn_model() -> tf.keras.Model:
    model = tf.keras.Sequential([tf.keras.Input(shape=(28, 28)), tf.keras.layers.Reshape(target_shape=(28, 28, 1)), tf.keras.layers.Conv2D(32, 3, activation='relu'), tf.keras.layers.Flatten(), tf.keras.layers.Dense(128, activation='relu'), tf.keras.layers.Dense(10)])
    return model