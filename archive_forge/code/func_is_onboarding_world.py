import logging
import math
import os
import pickle
import threading
import time
import uuid
import errno
import requests
from parlai.mturk.core.dev.agents import (
from parlai.mturk.core.dev.socket_manager import Packet, SocketManager
from parlai.mturk.core.dev.worker_manager import WorkerManager
from parlai.mturk.core.dev.mturk_data_handler import MTurkDataHandler
import parlai.mturk.core.dev.data_model as data_model
import parlai.mturk.core.dev.mturk_utils as mturk_utils
import parlai.mturk.core.dev.server_utils as server_utils
import parlai.mturk.core.dev.shared_utils as shared_utils
def is_onboarding_world(self, conversation_id):
    return conversation_id is not None and conversation_id.startswith('o_')