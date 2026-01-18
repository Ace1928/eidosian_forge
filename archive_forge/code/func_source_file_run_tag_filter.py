import json
from tensorboard.data import provider
from tensorboard.plugins.debugger_v2 import debug_data_multiplexer
def source_file_run_tag_filter(run, index):
    """Create a RunTagFilter for listing source files.

    Args:
      run: tfdbg2 run name.
      index: The index for the source file of which the content is to be
        accessed.

    Returns:
      `RunTagFilter` for accessing the content of the source file.
    """
    return provider.RunTagFilter(runs=[run], tags=['%s_%d' % (SOURCE_FILE_BLOB_TAG_PREFIX, index)])