from collections import deque
from http.server import HTTPServer, SimpleHTTPRequestHandler
import logging
import queue
from socketserver import ThreadingMixIn
import threading
import time
import traceback
from typing import List
import ray.cloudpickle as pickle
from ray.rllib.env.policy_client import (
from ray.rllib.offline.input_reader import InputReader
from ray.rllib.offline.io_context import IOContext
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override, PublicAPI
from ray.rllib.evaluation.metrics import RolloutMetrics
from ray.rllib.evaluation.sampler import SamplerInput
from ray.rllib.utils.typing import SampleBatchType
def report_data(data):
    nonlocal child_rollout_worker
    batch = data['samples']
    batch.decompress_if_needed()
    samples_queue.append(batch)
    if len(samples_queue) == samples_queue.maxlen:
        logger.warning('PolicyServerInput queue is full! Purging half of the samples (oldest).')
        for _ in range(samples_queue.maxlen // 2):
            samples_queue.popleft()
    for rollout_metric in data['metrics']:
        metrics_queue.put(rollout_metric)
    if child_rollout_worker is not None:
        child_rollout_worker.set_weights(rollout_worker.get_weights(), rollout_worker.get_global_vars())