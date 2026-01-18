import json
from tensorboard.data import provider
from tensorboard.plugins.debugger_v2 import debug_data_multiplexer
def execution_digest_run_tag_filter(run, begin, end):
    """Create a RunTagFilter for ExecutionDigests.

    This differs from `execution_data_run_tag_filter()` in that it is for
    the small-size digest objects for execution debug events, instead of the
    full-size data objects.

    Args:
      run: tfdbg2 run name.
      begin: Beginning index of ExecutionDigests.
      end: Ending index of ExecutionDigests.

    Returns:
      `RunTagFilter` for the run and range of ExecutionDigests.
    """
    return provider.RunTagFilter(runs=[run], tags=['%s_%d_%d' % (EXECUTION_DIGESTS_BLOB_TAG_PREFIX, begin, end)])