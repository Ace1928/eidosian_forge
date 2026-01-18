import torch
import numpy as np
from projects.controllable_dialogue.tasks.build import build
from .stopwords import STOPWORDS
from .nidf import load_word2nidf
from .arora import SentenceEmbedder, load_arora
def get_qn_bucket_probs():
    """
    Assuming we have 11 CT question buckets (0 to 10), compute P(bucket|question=1) and
    P(bucket|question=0); this is needed so we can probabilistically assign incoming
    training examples to buckets.

    Returns:
      prob_bucket_given_qn: list of floats length 11; P(bucket|question=1)
      prob_bucket_given_notqn: list of floats length 11; P(bucket|question=0)
    """
    prob_qn = 41101 / 131438
    prob_bucket_n = prob_qn / 5.5
    prob_bucket_0 = 1 - 10 * prob_bucket_n
    prob_bucket = [prob_bucket_0] + [prob_bucket_n] * 10
    prob_bucket_given_qn = [pb * (i / 10) / prob_qn for i, pb in enumerate(prob_bucket)]
    prob_bucket_given_notqn = [pb * ((10 - i) / 10) / (1 - prob_qn) for i, pb in enumerate(prob_bucket)]
    return (prob_bucket_given_qn, prob_bucket_given_notqn)