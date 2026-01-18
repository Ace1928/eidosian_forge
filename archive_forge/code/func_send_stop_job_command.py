import json
import os
import signal
from typing import TYPE_CHECKING, Any, Dict
from rq.exceptions import InvalidJobOperation
from rq.job import Job
def send_stop_job_command(connection: 'Redis', job_id: str, serializer=None):
    """
    Instruct a worker to stop a job

    Args:
        connection (Redis): A Redis Connection
        job_id (str): The Job ID
        serializer (): The serializer
    """
    job = Job.fetch(job_id, connection=connection, serializer=serializer)
    if not job.worker_name:
        raise InvalidJobOperation('Job is not currently executing')
    send_command(connection, job.worker_name, 'stop-job', job_id=job_id)