import copy
import os
import re
import traceback
import numpy as np
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.platform import gfile
def captions(self):
    return [item.caption for item in self._items]