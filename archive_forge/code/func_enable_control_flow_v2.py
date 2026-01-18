import os
import traceback
from tensorflow.python import tf2
from tensorflow.python.platform import tf_logging as logging
def enable_control_flow_v2():
    """Use control flow v2.

  Do not use this symbol. This will be removed.
  """
    global ENABLE_CONTROL_FLOW_V2
    ENABLE_CONTROL_FLOW_V2 = True