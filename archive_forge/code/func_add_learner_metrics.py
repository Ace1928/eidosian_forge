import copy
import queue
import threading
from typing import Dict, Optional
from ray.util.timer import _Timer
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.execution.minibatch_buffer import MinibatchBuffer
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.metrics.learner_info import LearnerInfoBuilder, LEARNER_INFO
from ray.rllib.utils.metrics.window_stat import WindowStat
from ray.util.iter import _NextValueNotReady
def add_learner_metrics(self, result: Dict, overwrite_learner_info=True) -> Dict:
    """Add internal metrics to a result dict."""

    def timer_to_ms(timer):
        return round(1000 * timer.mean, 3)
    if overwrite_learner_info:
        result['info'].update({'learner_queue': self.learner_queue_size.stats(), LEARNER_INFO: copy.deepcopy(self.learner_info), 'timing_breakdown': {'learner_grad_time_ms': timer_to_ms(self.grad_timer), 'learner_load_time_ms': timer_to_ms(self.load_timer), 'learner_load_wait_time_ms': timer_to_ms(self.load_wait_timer), 'learner_dequeue_time_ms': timer_to_ms(self.queue_timer)}})
    else:
        result['info'].update({'learner_queue': self.learner_queue_size.stats(), 'timing_breakdown': {'learner_grad_time_ms': timer_to_ms(self.grad_timer), 'learner_load_time_ms': timer_to_ms(self.load_timer), 'learner_load_wait_time_ms': timer_to_ms(self.load_wait_timer), 'learner_dequeue_time_ms': timer_to_ms(self.queue_timer)}})
    return result