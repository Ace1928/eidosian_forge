import json
from tensorboard.data import provider
from tensorboard.plugins.debugger_v2 import debug_data_multiplexer
def source_file_list_run_tag_filter(run):
    """Create a RunTagFilter for listing source files.

    Args:
      run: tfdbg2 run name.

    Returns:
      `RunTagFilter` for listing the source files in the tfdbg2 run.
    """
    return provider.RunTagFilter(runs=[run], tags=[SOURCE_FILE_LIST_BLOB_TAG])