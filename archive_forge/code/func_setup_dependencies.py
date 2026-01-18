import logging
import sys
import traceback
import uuid
import warnings
from collections import namedtuple
from datetime import datetime, timedelta, timezone
from functools import total_ordering
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, Union
from redis import WatchError
from .timeouts import BaseDeathPenalty, UnixSignalDeathPenalty
from .connections import resolve_connection
from .defaults import DEFAULT_RESULT_TTL
from .dependency import Dependency
from .exceptions import DequeueTimeout, NoSuchJobError
from .job import Callback, Job, JobStatus
from .logutils import blue, green
from .serializers import resolve_serializer
from .types import FunctionReferenceType, JobDependencyType
from .utils import as_text, backend_class, compact, get_version, import_attribute, parse_timeout, utcnow
def setup_dependencies(self, job: 'Job', pipeline: Optional['Pipeline']=None) -> 'Job':
    """If a _dependent_ job depends on any unfinished job, register all the
        _dependent_ job's dependencies instead of enqueueing it.

        `Job#fetch_dependencies` sets WATCH on all dependencies. If
        WatchError is raised in the when the pipeline is executed, that means
        something else has modified either the set of dependencies or the
        status of one of them. In this case, we simply retry.

        Args:
            job (Job): The job
            pipeline (Optional[Pipeline], optional): The Redis Pipeline. Defaults to None.

        Returns:
            job (Job): The Job
        """
    if len(job._dependency_ids) > 0:
        orig_status = job.get_status(refresh=False)
        pipe = pipeline if pipeline is not None else self.connection.pipeline()
        while True:
            try:
                pipe.watch(job.dependencies_key)
                dependencies = job.fetch_dependencies(watch=True, pipeline=pipe)
                pipe.multi()
                for dependency in dependencies:
                    if dependency.get_status(refresh=False) != JobStatus.FINISHED:
                        job.set_status(JobStatus.DEFERRED, pipeline=pipe)
                        job.register_dependency(pipeline=pipe)
                        job.save(pipeline=pipe)
                        job.cleanup(ttl=job.ttl, pipeline=pipe)
                        if pipeline is None:
                            pipe.execute()
                        return job
                break
            except WatchError:
                if pipeline is None:
                    job._status = orig_status
                    continue
                else:
                    raise
    elif pipeline is not None:
        pipeline.multi()
    return job