import os
import sys
import uuid
import traceback
import json
import param
from ._version import __version__
@classmethod
def add_server_delete_action(cls, action):
    cls._server_delete_actions.append(action)