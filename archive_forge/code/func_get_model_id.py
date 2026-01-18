import threading
import time
from collections import Counter
import numpy
import pytest
from thinc.api import (
from thinc.compat import has_cupy_gpu
from ..util import make_tempdir
def get_model_id(id_list, index):
    id_list.append(create_model(name=f'worker{index}').id)