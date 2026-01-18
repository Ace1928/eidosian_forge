from unittest.mock import patch
import pytest
import torch
import os
import shutil
from llama_recipes.utils.train_utils import train
@pytest.fixture(scope='session')
def temp_output_dir():
    temp_output_dir = 'tmp'
    os.mkdir(os.path.join(os.getcwd(), temp_output_dir))
    yield temp_output_dir
    shutil.rmtree(temp_output_dir)