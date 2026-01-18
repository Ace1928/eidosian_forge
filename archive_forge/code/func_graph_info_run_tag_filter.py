import json
from tensorboard.data import provider
from tensorboard.plugins.debugger_v2 import debug_data_multiplexer
def graph_info_run_tag_filter(run, graph_id):
    """Create a RunTagFilter for graph info.

    Args:
      run: tfdbg2 run name.
      graph_id: Debugger-generated ID of the graph in question.

    Returns:
      `RunTagFilter` for the run and range of graph info.
    """
    if not graph_id:
        raise ValueError('graph_id must not be None or empty.')
    return provider.RunTagFilter(runs=[run], tags=['%s_%s' % (GRAPH_INFO_BLOB_TAG_PREFIX, graph_id)])