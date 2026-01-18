import os
from os.path import join
import shutil
import time
import uuid
from lion_pytorch import Lion
import pytest
import torch
import bitsandbytes as bnb
import bitsandbytes.functional as F
from tests.helpers import describe_dtype, id_formatter
def assert_most_approx_close(a, b, rtol=0.001, atol=0.001, max_error_count=0):
    idx = torch.isclose(a, b, rtol=rtol, atol=atol)
    error_count = (idx == 0).sum().item()
    if error_count > max_error_count:
        print(f'Too many values not close: assert {error_count} < {max_error_count}')
        torch.testing.assert_close(a, b, rtol=rtol, atol=atol)