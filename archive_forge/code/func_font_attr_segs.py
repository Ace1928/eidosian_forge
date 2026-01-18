import copy
import os
import re
import traceback
import numpy as np
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.platform import gfile
@property
def font_attr_segs(self):
    return self._font_attr_segs