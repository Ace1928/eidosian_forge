import torch
import numpy as np
from projects.controllable_dialogue.tasks.build import build
from .stopwords import STOPWORDS
from .nidf import load_word2nidf
from .arora import SentenceEmbedder, load_arora
def get_ctrl_vec(exs, history, control_settings):
    """
    Given a batch of examples with given history, return the bucketed CT control values.
    This is used both when training and evaluating CT systems.

    Inputs:
      exs: list length batch_size of message dictionaries. Each dictionary contains
        a 'text' field, and a field for each CT control we're using, along with the
        value of the CT control variable.
      history: list length batch_size of ConvAI2History objects. These represent the
        conversation history.
      control_settings: dictionary containing info about CT controls.
        See ControllableSeq2seqAgent.control_settings.

    Returns:
      ctrl_vec: torch Tensor shape (batch_size, num_controls), with the bucketed values
        for the CT controls we're using. If there's no CT controls, return None.
    """
    if len(control_settings) == 0:
        return None
    ctrl_vec = -torch.ones((len(exs), len(control_settings))).long()
    for batch_idx, ex in enumerate(exs):
        for ctrl, ctrl_info in control_settings.items():
            set_val = ctrl_info['set_value']
            if set_val is not None:
                bucket = set_val
            else:
                if ctrl not in ex:
                    raise ValueError("The CT control '%s' is not present as a key in the message dictionary:\n%s\nIf training a CT model, perhaps your training data is missing the annotations. If talking interactively, perhaps you forgot to set --set-controls." % (ctrl, str(ex)))
                num_buckets = ctrl_info['num_buckets']
                bucketing_fn = CONTROL2BUCKETINGFN[ctrl]
                bucket = bucketing_fn(ex, ctrl, num_buckets)
            ctrl_idx = ctrl_info['idx']
            ctrl_vec[batch_idx, ctrl_idx] = bucket
    return ctrl_vec