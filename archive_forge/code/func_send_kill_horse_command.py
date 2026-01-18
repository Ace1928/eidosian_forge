import json
import os
import signal
from typing import TYPE_CHECKING, Any, Dict
from rq.exceptions import InvalidJobOperation
from rq.job import Job
def send_kill_horse_command(connection: 'Redis', worker_name: str):
    """
    Tell worker to kill it's horse

    Args:
        connection (Redis): A Redis Connection
        worker_name (str): The Job ID
    """
    send_command(connection, worker_name, 'kill-horse')