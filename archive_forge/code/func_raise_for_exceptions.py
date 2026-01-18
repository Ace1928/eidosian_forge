from __future__ import annotations
import contextlib
from lazyops.types import BaseModel, Field, root_validator
from lazyops.types.models import ConfigDict, schema_extra
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException
from kvdb.types.jobs import Job, JobStatus
from lazyops.libs.logging import logger
from typing import Any, Dict, List, Optional, Type, TypeVar, Literal, Union, Set, TYPE_CHECKING
def raise_for_exceptions(self):
    """
        Helper method to raise for exceptions
        """
    if not self.has_error:
        return
    raise HTTPException(status_code=self.status_code, detail=f'{self.error}')