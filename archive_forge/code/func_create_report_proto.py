import collections
import hashlib
import os
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import tensor_tracer_pb2
def create_report_proto(self, tt_config, tt_parameters, tensor_trace_order, tensor_trace_points, collected_signature_types):
    """Creates and returns a proto that stores tensor tracer configuration.

    Args:
      tt_config: TensorTracerConfig object holding information about the run
        environment (device, # cores, # hosts), and tensor tracer version
        information.
      tt_parameters: TTParameters objects storing the user provided parameters
        for tensor tracer.
      tensor_trace_order: TensorTraceOrder object storing a topological order of
        the graph.
      tensor_trace_points: Progromatically added trace_points/checkpoints.
      collected_signature_types: The signature types collected, e,g, norm,
        max, min, mean...
    Returns:
      TensorTracerReport proto.
    """
    report = tensor_tracer_pb2.TensorTracerReport()
    report.config.version = tt_config.version
    report.config.device = tt_config.device_type
    report.config.num_cores = tt_config.num_replicas
    report.config.num_hosts = tt_config.num_hosts
    report.config.num_cores_per_host = tt_config.num_replicas_per_host
    report.config.submode = tt_parameters.submode
    report.config.trace_mode = tt_parameters.trace_mode
    for signature_name, _ in sorted(collected_signature_types.items(), key=lambda x: x[1]):
        report.config.signatures.append(signature_name)
    for tensor in tensor_trace_order.graph_order.tensors:
        tensor_def = tensor_tracer_pb2.TensorTracerReport.TracedTensorDef()
        tensor_def.name = tensor.name
        if tensor.name in tensor_trace_order.tensorname_to_cache_idx:
            tensor_def.is_traced = True
            tensor_def.cache_index = tensor_trace_order.tensorname_to_cache_idx[tensor.name]
        else:
            if tt_parameters.use_fingerprint_subdir:
                continue
            tensor_def.is_traced = False
        if tensor.name in tensor_trace_points:
            tensor_def.trace_point_name = tensor_trace_points[tensor.name]
        if tensor.name in self.instrument_records:
            tensor_def.explanation = self.instrument_records[tensor.name]
        elif tensor.op.name in self.instrument_records:
            tensor_def.explanation = self.instrument_records[tensor.op.name]
        report.tensordef[tensor.name].CopyFrom(tensor_def)
    report.fingerprint = proto_fingerprint(report)
    logging.info('TensorTracerProto fingerprint is %s.', report.fingerprint)
    tf_graph = tensor_trace_order.graph_order.graph
    report.graphdef.CopyFrom(tf_graph.as_graph_def())
    return report