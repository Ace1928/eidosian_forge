import os
import math
import random
from pathlib import Path
import pytest
from IPython.utils import text
def test_full_eval_formatter():
    f = text.FullEvalFormatter()
    eval_formatter_check(f)
    eval_formatter_slicing_check(f)