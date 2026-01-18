from __future__ import print_function
import mido
import numpy as np
import math
import warnings
import collections
import copy
import functools
import six
from heapq import merge
from .instrument import Instrument
from .containers import (KeySignature, TimeSignature, Lyric, Note,
from .utilities import (key_name_to_key_number, qpm_to_bpm)
def estimate_tempi(self):
    """Return an empirical estimate of tempos and each tempo's probability.
        Based on "Automatic Extraction of Tempo and Beat from Expressive
        Performance", Dixon 2001.

        Returns
        -------
        tempos : np.ndarray
            Array of estimated tempos, in beats per minute.
        probabilities : np.ndarray
            Array of the probabilities of each tempo estimate.

        """
    onsets = self.get_onsets()
    ioi = np.diff(onsets)
    ioi = ioi[ioi > 0.05]
    ioi = ioi[ioi < 2]
    for n in range(ioi.shape[0]):
        while ioi[n] < 0.2:
            ioi[n] *= 2
    clusters = np.array([])
    cluster_counts = np.array([])
    for interval in ioi:
        if (np.abs(clusters - interval) < 0.025).any():
            k = np.argmin(clusters - interval)
            clusters[k] = (cluster_counts[k] * clusters[k] + interval) / (cluster_counts[k] + 1)
            cluster_counts[k] += 1
        else:
            clusters = np.append(clusters, interval)
            cluster_counts = np.append(cluster_counts, 1.0)
    cluster_sort = np.argsort(cluster_counts)[::-1]
    clusters = clusters[cluster_sort]
    cluster_counts = cluster_counts[cluster_sort]
    cluster_counts /= cluster_counts.sum()
    return (60.0 / clusters, cluster_counts)