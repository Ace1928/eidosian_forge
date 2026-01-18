from __future__ import annotations
import aifc
import audioop
import base64
import collections
import hashlib
import hmac
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import wave
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from .audio import AudioData, get_flac_converter
from .exceptions import (
def recognize_tensorflow(self, audio_data, tensor_graph='tensorflow-data/conv_actions_frozen.pb', tensor_label='tensorflow-data/conv_actions_labels.txt'):
    """
        Performs speech recognition on ``audio_data`` (an ``AudioData`` instance).

        Path to Tensor loaded from ``tensor_graph``. You can download a model here: http://download.tensorflow.org/models/speech_commands_v0.01.zip

        Path to Tensor Labels file loaded from ``tensor_label``.
        """
    assert isinstance(audio_data, AudioData), 'Data must be audio data'
    assert isinstance(tensor_graph, str), '``tensor_graph`` must be a string'
    assert isinstance(tensor_label, str), '``tensor_label`` must be a string'
    try:
        import tensorflow as tf
    except ImportError:
        raise RequestError('missing tensorflow module: ensure that tensorflow is set up correctly.')
    if not tensor_graph == self.lasttfgraph:
        self.lasttfgraph = tensor_graph
        with tf.gfile.FastGFile(tensor_graph, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
        self.tflabels = [line.rstrip() for line in tf.gfile.GFile(tensor_label)]
    wav_data = audio_data.get_wav_data(convert_rate=16000, convert_width=2)
    with tf.Session() as sess:
        input_layer_name = 'wav_data:0'
        output_layer_name = 'labels_softmax:0'
        softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)
        predictions, = sess.run(softmax_tensor, {input_layer_name: wav_data})
        top_k = predictions.argsort()[-1:][::-1]
        for node_id in top_k:
            human_string = self.tflabels[node_id]
            return human_string