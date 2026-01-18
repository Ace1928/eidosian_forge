import json
from tensorboard.data import provider
from tensorboard.plugins.debugger_v2 import debug_data_multiplexer
def stack_frames_run_tag_filter(run, stack_frame_ids):
    """Create a RunTagFilter for querying stack frames.

    Args:
      run: tfdbg2 run name.
      stack_frame_ids: The stack_frame_ids being requested.

    Returns:
      `RunTagFilter` for accessing the content of the source file.
    """
    return provider.RunTagFilter(runs=[run], tags=[STACK_FRAMES_BLOB_TAG_PREFIX + '_' + '_'.join(stack_frame_ids)])