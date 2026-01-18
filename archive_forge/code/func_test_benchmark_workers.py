import unittest
from pygame.threads import FuncResult, tmap, WorkerQueue, Empty, STOP
from pygame import threads, Surface, transform
import time
def test_benchmark_workers(self):
    """Ensure benchmark_workers performance measure functions properly with both default and specified inputs"""
    'tags:long_running'
    optimal_workers = threads.benchmark_workers()
    self.assertIsInstance(optimal_workers, int)
    self.assertTrue(0 <= optimal_workers < 64)

    def smooth_scale_bench(data):
        transform.smoothscale(data, (128, 128))
    surf_data = [Surface((x, x), 0, 32) for x in range(12, 64, 12)]
    best_num_workers = threads.benchmark_workers(smooth_scale_bench, surf_data)
    self.assertIsInstance(best_num_workers, int)