import asyncio
import inspect
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import List, Union
from unittest import mock
import torch
import accelerate
from ..state import AcceleratorState, PartialState
from ..utils import (
def are_the_same_tensors(tensor):
    state = AcceleratorState()
    tensor = tensor[None].clone().to(state.device)
    tensors = gather(tensor).cpu()
    tensor = tensor[0].cpu()
    for i in range(tensors.shape[0]):
        if not torch.equal(tensors[i], tensor):
            return False
    return True