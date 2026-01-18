import os
import sys
from io import BytesIO
from types import CodeType, FunctionType
import dill
from packaging import version
from .. import config
def create_torchGenerator(state):
    generator = torch.Generator()
    generator.set_state(state)
    return generator