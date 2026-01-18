import torch._dynamo.test_case
import unittest.mock
import os
import contextlib
import torch._logging
import torch._logging._internal
from torch._dynamo.utils import LazyString
import logging
def new_emit(record):
    old_emit(record)
    emit_post_hook(record)