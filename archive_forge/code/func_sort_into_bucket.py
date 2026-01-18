import torch
import numpy as np
from projects.controllable_dialogue.tasks.build import build
from .stopwords import STOPWORDS
from .nidf import load_word2nidf
from .arora import SentenceEmbedder, load_arora
def sort_into_bucket(val, bucket_lbs):
    """
    Returns the highest bucket such that val >= lower bound for that bucket.

    Inputs:
      val: float. The value to be sorted into a bucket.
      bucket_lbs: list of floats, sorted ascending.

    Returns:
      bucket_id: int in range(num_buckets); the bucket that val belongs to.
    """
    num_buckets = len(bucket_lbs)
    for bucket_id in range(num_buckets - 1, -1, -1):
        lb = bucket_lbs[bucket_id]
        if val >= lb:
            return bucket_id
    raise ValueError('val %f is not >= any of the lower bounds: %s' % (val, bucket_lbs))