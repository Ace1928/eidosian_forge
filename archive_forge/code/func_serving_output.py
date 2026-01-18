from __future__ import annotations
import random
from typing import Optional, Tuple, Union
import numpy as np
import tensorflow as tf
from ...activations_tf import get_tf_activation, glu
from ...modeling_tf_outputs import (
from ...modeling_tf_utils import (
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
from .configuration_speech_to_text import Speech2TextConfig
def serving_output(self, output):
    pkv = tf.tuple(output.past_key_values)[1] if self.config.use_cache else None
    dec_hs = tf.convert_to_tensor(output.decoder_hidden_states) if self.config.output_hidden_states else None
    dec_attns = tf.convert_to_tensor(output.decoder_attentions) if self.config.output_attentions else None
    cross_attns = tf.convert_to_tensor(output.cross_attentions) if self.config.output_attentions else None
    enc_hs = tf.convert_to_tensor(output.encoder_hidden_states) if self.config.output_hidden_states else None
    enc_attns = tf.convert_to_tensor(output.encoder_attentions) if self.config.output_attentions else None
    return TFSeq2SeqLMOutput(logits=output.logits, past_key_values=pkv, decoder_hidden_states=dec_hs, decoder_attentions=dec_attns, cross_attentions=cross_attns, encoder_last_hidden_state=output.encoder_last_hidden_state, encoder_hidden_states=enc_hs, encoder_attentions=enc_attns)