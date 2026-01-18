import collections
import hashlib
import os
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import tensor_tracer_pb2
def create_report(self, tt_config, tt_parameters, tensor_trace_order, tensor_trace_points):
    """Creates a report file and writes the trace information."""
    with OpenReportFile(tt_parameters) as self._report_file:
        self._write_config_section(tt_config, tt_parameters)
        self._write_op_list_section(tensor_trace_order.graph_order)
        self._write_tensor_list_section(tensor_trace_order.graph_order)
        self._write_trace_points(tensor_trace_points)
        self._write_cache_index_map_section(tensor_trace_order)
        self._write_reason_section()
        self._write_graph_section(tensor_trace_order.graph_order)