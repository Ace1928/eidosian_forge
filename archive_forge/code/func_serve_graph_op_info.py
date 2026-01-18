import threading
from werkzeug import wrappers
from tensorboard import errors
from tensorboard import plugin_util
from tensorboard.plugins import base_plugin
from tensorboard.plugins.debugger_v2 import debug_data_provider
from tensorboard.backend import http_util
@wrappers.Request.application
def serve_graph_op_info(self, request):
    """Serve information for ops in graphs.

        The request specifies the op name and the ID of the graph that
        contains the op.

        The response contains a JSON object with the following fields:
          - op_type
          - op_name
          - graph_ids: Stack of graph IDs that the op is located in, from
            outermost to innermost. The length of this array is always >= 1.
            The length is 1 if and only if the graph is an outermost graph.
          - num_outputs: Number of output tensors.
          - output_tensor_ids: The debugger-generated number IDs for the
            symbolic output tensors of the op (an array of numbers).
          - host_name: Name of the host on which the op is created.
          - stack_trace: Stack frames of the op's creation.
          - inputs: Specifications of all inputs to this op.
            Currently only immediate (one level of) inputs are provided.
            This is an array of length N_in, where N_in is the number of
            data inputs received by the op. Each element of the array is an
            object with the following fields:
              - op_name: Name of the op that provides the input tensor.
              - output_slot: 0-based output slot index from which the input
                tensor emits.
              - data: A recursive data structure of this same schema.
                This field is not populated (undefined) at the leaf nodes
                of this recursive data structure.
                In the rare case wherein the data for an input cannot be
                retrieved properly (e.g., special internal op types), this
                field will be unpopulated.
            This is an empty list for an op with no inputs.
          - consumers: Specifications for all the downstream consuming ops of
            this. Currently only immediate (one level of) consumers are provided.
            This is an array of length N_out, where N_out is the number of
            symbolic tensors output by this op.
            Each element of the array is an array of which the length equals
            the number of downstream ops that consume the corresponding symbolic
            tensor (only data edges are tracked).
            Each element of the array is an object with the following fields:
              - op_name: Name of the op that receives the output tensor as an
                input.
              - input_slot: 0-based input slot index at which the downstream
                op receives this output tensor.
              - data: A recursive data structure of this very schema.
                This field is not populated (undefined) at the leaf nodes
                of this recursive data structure.
                In the rare case wherein the data for a consumer op cannot be
                retrieved properly (e.g., special internal op types), this
                field will be unpopulated.
            If this op has no output tensors, this is an empty array.
            If one of the output tensors of this op has no consumers, the
            corresponding element is an empty array.
        """
    experiment = plugin_util.experiment_id(request.environ)
    run = request.args.get('run')
    if run is None:
        return _missing_run_error_response(request)
    graph_id = request.args.get('graph_id')
    op_name = request.args.get('op_name')
    run_tag_filter = debug_data_provider.graph_op_info_run_tag_filter(run, graph_id, op_name)
    blob_sequences = self._data_provider.read_blob_sequences(experiment_id=experiment, plugin_name=self.plugin_name, run_tag_filter=run_tag_filter)
    tag = next(iter(run_tag_filter.tags))
    try:
        return http_util.Respond(request, self._data_provider.read_blob(blob_key=blob_sequences[run][tag][0].blob_key), 'application/json')
    except errors.NotFoundError as e:
        return _error_response(request, str(e))