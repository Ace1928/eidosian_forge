import json
from tensorboard.data import provider
from tensorboard.plugins.debugger_v2 import debug_data_multiplexer
def execution_data_run_tag_filter(run, begin, end):
    """Create a RunTagFilter for Execution data objects.

    This differs from `execution_digest_run_tag_filter()` in that it is
    for the detailed data objects for execution, instead of the digests.

    Args:
      run: tfdbg2 run name.
      begin: Beginning index of Execution.
      end: Ending index of Execution.

    Returns:
      `RunTagFilter` for the run and range of ExecutionDigests.
    """
    return provider.RunTagFilter(runs=[run], tags=['%s_%d_%d' % (EXECUTION_DATA_BLOB_TAG_PREFIX, begin, end)])