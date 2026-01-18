import os
from typing import List, Optional, Union
from tensorflow.python.eager import context
from tensorflow.python.framework import config as tf_config
from tensorflow.python.framework import device as tf_device
from tensorflow.python.util.tf_export import tf_export
@tf_export('experimental.dtensor.jobs', v1=[])
def jobs() -> List[str]:
    """Returns a list of job names of all clients in this DTensor cluster."""
    d_jobs = os.environ.get(_DT_JOBS)
    if d_jobs is None:
        return []
    d_jobs_list = d_jobs.split(',')
    if any([name.startswith('/bns/') for name in d_jobs_list]):
        if d_jobs_list != sorted(d_jobs_list, key=_bns_task_id):
            raise ValueError(f'Unexpected DTENSOR_JOBS content {d_jobs}. Sort entries in DTENSOR_JOBS because cluster construction relies on the order.')
    return d_jobs_list