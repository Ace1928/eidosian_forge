import json
import logging
import os
import pickle
import sqlite3
import time
import threading
import parlai.mturk.core.dev.shared_utils as shared_utils
from parlai.mturk.core.dev.agents import AssignState
def force_dir(path):
    """
    Make sure the parent dir exists for path so we can write a file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)