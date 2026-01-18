import math
import numpy as np
from tensorflow.python.debug.cli import command_parser
from tensorflow.python.debug.cli import debugger_cli_common
from tensorflow.python.debug.cli import tensor_format
from tensorflow.python.debug.lib import common
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
def get_run_short_description(run_call_count, fetches, feed_dict, is_callable_runner=False):
    """Get a short description of the run() call.

  Args:
    run_call_count: (int) Run call counter.
    fetches: Fetches of the `Session.run()` call. See doc of `Session.run()`
      for more details.
    feed_dict: Feeds to the `Session.run()` call. See doc of `Session.run()`
      for more details.
    is_callable_runner: (bool) whether a runner returned by
        Session.make_callable is being run.

  Returns:
    (str) A short description of the run() call, including information about
      the fetche(s) and feed(s).
  """
    if is_callable_runner:
        return 'runner from make_callable()'
    description = 'run #%d: ' % run_call_count
    if isinstance(fetches, (tensor_lib.Tensor, ops.Operation, variables.Variable)):
        description += '1 fetch (%s); ' % common.get_graph_element_name(fetches)
    else:
        num_fetches = len(common.get_flattened_names(fetches))
        if num_fetches > 1:
            description += '%d fetches; ' % num_fetches
        else:
            description += '%d fetch; ' % num_fetches
    if not feed_dict:
        description += '0 feeds'
    elif len(feed_dict) == 1:
        for key in feed_dict:
            description += '1 feed (%s)' % (key if isinstance(key, str) or not hasattr(key, 'name') else key.name)
    else:
        description += '%d feeds' % len(feed_dict)
    return description