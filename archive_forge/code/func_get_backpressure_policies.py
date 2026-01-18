from typing import TYPE_CHECKING
import ray
from .backpressure_policy import BackpressurePolicy
from .concurrency_cap_backpressure_policy import ConcurrencyCapBackpressurePolicy
from .streaming_output_backpressure_policy import StreamingOutputBackpressurePolicy
def get_backpressure_policies(topology: 'Topology'):
    data_context = ray.data.DataContext.get_current()
    policies = data_context.get_config(ENABLED_BACKPRESSURE_POLICIES_CONFIG_KEY, ENABLED_BACKPRESSURE_POLICIES)
    return [policy(topology) for policy in policies]