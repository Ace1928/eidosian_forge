import os
from pytest import raises
from accelerate import PartialState, notebook_launcher
from accelerate.test_utils import require_bnb
from accelerate.utils import is_bnb_available
def test_can_initialize():
    notebook_launcher(basic_function, (), num_processes=NUM_PROCESSES)