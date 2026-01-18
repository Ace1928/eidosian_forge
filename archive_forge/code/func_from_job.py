from __future__ import annotations
import contextlib
from lazyops.types import BaseModel, Field, root_validator
from lazyops.types.models import ConfigDict, schema_extra
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException
from kvdb.types.jobs import Job, JobStatus
from lazyops.libs.logging import logger
from typing import Any, Dict, List, Optional, Type, TypeVar, Literal, Union, Set, TYPE_CHECKING
@classmethod
def from_job(cls, job: Job, callback_id: Optional[str]=None, **kwargs) -> 'JobResult':
    """
        Returns the JobResult from the Job
        """
    return cls(job_id=job.id, status=job.status, progress=job.progress.completed, duration=job.duration / 1000 if job.duration else None, callback_id=callback_id, **kwargs)