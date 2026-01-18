import inspect
import logging
import numpy as np
import sys
from ray.util.client.ray_client_helpers import ray_start_client_server
from ray._private.ray_microbenchmark_helpers import timeit
def benchmark_simple_actor(ray, results):

    @ray.remote(num_cpus=0)
    class Actor:

        def small_value(self):
            return b'ok'

        def small_value_arg(self, x):
            return b'ok'

        def small_value_batch(self, n):
            ray.get([self.small_value.remote() for _ in range(n)])
    a = Actor.remote()

    def actor_sync():
        ray.get(a.small_value.remote())
    results += timeit('client: 1:1 actor calls sync', actor_sync)

    def actor_async():
        ray.get([a.small_value.remote() for _ in range(1000)])
    results += timeit('client: 1:1 actor calls async', actor_async, 1000)
    a = Actor.options(max_concurrency=16).remote()

    def actor_concurrent():
        ray.get([a.small_value.remote() for _ in range(1000)])
    results += timeit('client: 1:1 actor calls concurrent', actor_concurrent, 1000)