import argparse
import importlib
import os
import sys as _sys
import datetime
import parlai
import parlai.utils.logging as logging
from parlai.core.build_data import modelzoo_path
from parlai.core.loader import (
from parlai.tasks.tasks import ids_to_tasks
from parlai.core.opt import Opt
from typing import List, Optional
def add_websockets_args(self):
    """
        Add websocket arguments.
        """
    self.add_chatservice_args()
    websockets = self.add_argument_group('Websockets')
    websockets.add_argument('--port', default=35496, type=int, help='Port to run the websocket handler')