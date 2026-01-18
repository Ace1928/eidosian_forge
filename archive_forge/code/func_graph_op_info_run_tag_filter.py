import json
from tensorboard.data import provider
from tensorboard.plugins.debugger_v2 import debug_data_multiplexer
def graph_op_info_run_tag_filter(run, graph_id, op_name):
    """Create a RunTagFilter for graph op info.

    Args:
      run: tfdbg2 run name.
      graph_id: Debugger-generated ID of the graph. This is assumed to
        be the ID of the graph that immediately encloses the op in question.
      op_name: Name of the op in question. (e.g., "Dense_1/MatMul")

    Returns:
      `RunTagFilter` for the run and range of graph op info.
    """
    if not graph_id:
        raise ValueError('graph_id must not be None or empty.')
    return provider.RunTagFilter(runs=[run], tags=['%s_%s_%s' % (GRAPH_OP_INFO_BLOB_TAG_PREFIX, graph_id, op_name)])