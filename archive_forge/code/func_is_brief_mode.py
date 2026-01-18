import os
import os.path
import re
from absl import flags
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
def is_brief_mode(self):
    return self.submode == _SUBMODE_BRIEF