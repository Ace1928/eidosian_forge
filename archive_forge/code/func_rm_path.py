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
def rm_path(path):
    shutil.rmtree(path)