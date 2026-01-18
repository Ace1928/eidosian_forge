import numpy as np
import tensorflow.compat.v2 as tf
from keras.src.utils import dataset_utils
from tensorflow.python.util.tf_export import keras_export
def read_and_decode_audio(path, sampling_rate=None, output_sequence_length=None):
    """Reads and decodes audio file."""
    audio = tf.io.read_file(path)
    if output_sequence_length is None:
        output_sequence_length = -1
    audio, default_audio_rate = tf.audio.decode_wav(contents=audio, desired_samples=output_sequence_length)
    if sampling_rate is not None:
        default_audio_rate = tf.cast(default_audio_rate, tf.int64)
        audio = tfio.audio.resample(input=audio, rate_in=default_audio_rate, rate_out=sampling_rate)
    return audio