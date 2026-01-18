import logging
import os
import sys
import json
def gcs_actor_scheduling_enabled():
    return os.environ.get('RAY_gcs_actor_scheduling_enabled') == 'true'