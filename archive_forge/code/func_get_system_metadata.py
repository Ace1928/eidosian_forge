import argparse
import json
import os
import socket
import time
import warnings
from pathlib import Path
from typing import Dict, List, Union
from zipfile import ZipFile
import numpy as np
import torch
from huggingface_hub.hf_api import list_models
from torch import nn
from tqdm import tqdm
from transformers import MarianConfig, MarianMTModel, MarianTokenizer
def get_system_metadata(repo_root):
    import git
    return {'helsinki_git_sha': git.Repo(path=repo_root, search_parent_directories=True).head.object.hexsha, 'transformers_git_sha': git.Repo(path='.', search_parent_directories=True).head.object.hexsha, 'port_machine': socket.gethostname(), 'port_time': time.strftime('%Y-%m-%d-%H:%M')}