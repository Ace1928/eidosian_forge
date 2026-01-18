import numpy as np
from tensorflow.core.protobuf import debug_event_pb2
def on_graph_execution_trace(self, graph_execution_trace_index, graph_execution_trace):
    """Monitor method for GraphExecutionTrace data object."""
    if self._limit > 0 and len(self._alerts) >= self._limit:
        return
    if graph_execution_trace.tensor_debug_mode == debug_event_pb2.TensorDebugMode.FULL_TENSOR:
        tensor_value = self._debug_data_reader.graph_execution_trace_to_tensor_value(graph_execution_trace)
        self._check_full_tensor_value(tensor_value, graph_execution_trace.wall_time, graph_execution_trace.op_type, graph_execution_trace.output_slot, graph_execution_trace_index=graph_execution_trace_index)
    elif graph_execution_trace.debug_tensor_value:
        self._check_debug_tensor_value(graph_execution_trace.tensor_debug_mode, graph_execution_trace.debug_tensor_value, graph_execution_trace.wall_time, graph_execution_trace.op_type, graph_execution_trace.output_slot, graph_execution_trace_index=graph_execution_trace_index)