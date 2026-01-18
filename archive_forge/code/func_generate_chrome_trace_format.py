import collections
import copy
import json
import re
from tensorflow.python.platform import build_info
from tensorflow.python.platform import tf_logging as logging
def generate_chrome_trace_format(self, show_dataflow=True, show_memory=False, op_time='schedule'):
    """Produces a trace in Chrome Trace Format.

    Args:
      show_dataflow: (Optional.) If True, add flow events to the trace
        connecting producers and consumers of tensors.
      show_memory: (Optional.) If True, add object snapshot events to the trace
        showing the sizes and lifetimes of tensors.
      op_time: (Optional.) How the execution time of op is shown in timeline.
        Possible values are "schedule", "gpu" and "all".
        "schedule" will show op from the time it is scheduled to the end of
          the scheduling.
          Notice by the end of its scheduling its async kernels may not start
          yet. It is shown using the default value from step_stats.
        "gpu" will show op with the execution time of its kernels on GPU.
        "all" will show op from the start of its scheduling to the end of
          its last kernel.

    Returns:
      A JSON formatted string in Chrome Trace format.
    """
    step_stats_analysis = self.analyze_step_stats(show_dataflow=show_dataflow, show_memory=show_memory, op_time=op_time)
    return step_stats_analysis.chrome_trace.format_to_string(pretty=True)