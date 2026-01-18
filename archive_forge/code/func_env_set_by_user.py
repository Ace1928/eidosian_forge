import logging
import os
import sys
import json
def env_set_by_user(key):
    return key in os.environ