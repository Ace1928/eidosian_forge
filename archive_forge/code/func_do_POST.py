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
def do_POST(self):
    content_len = int(self.headers.get('Content-Length'), 0)
    raw_body = self.rfile.read(content_len)
    parsed_input = pickle.loads(raw_body)
    try:
        response = self.execute_command(parsed_input)
        self.send_response(200)
        self.end_headers()
        self.wfile.write(pickle.dumps(response))
    except Exception:
        self.send_error(500, traceback.format_exc())