import zlib
from base64 import b64decode, b64encode
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from redis import Redis
from .defaults import UNSERIALIZABLE_RETURN_VALUE_PAYLOAD
from .job import Job
from .serializers import resolve_serializer
from .utils import decode_redis_hash, now
@classmethod
def fetch_latest(cls, job: Job, serializer=None, timeout: int=0) -> Optional['Result']:
    """Returns the latest result for given job instance or ID.

        If a non-zero timeout is provided, block for a result until timeout is reached.
        """
    if timeout:
        timeout_ms = timeout * 1000
        response = job.connection.xread({cls.get_key(job.id): '0-0'}, block=timeout_ms)
        if not response:
            return None
        response = response[0]
        response = response[1]
        result_id, payload = response[-1]
    else:
        response = job.connection.xrevrange(cls.get_key(job.id), '+', '-', count=1)
        if not response:
            return None
        result_id, payload = response[0]
    res = cls.restore(job.id, result_id.decode(), payload, connection=job.connection, serializer=serializer)
    return res