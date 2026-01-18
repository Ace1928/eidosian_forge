from __future__ import annotations
import contextlib
from lazyops.types import BaseModel, Field, root_validator
from lazyops.types.models import ConfigDict, schema_extra
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException
from kvdb.types.jobs import Job, JobStatus
from lazyops.libs.logging import logger
from typing import Any, Dict, List, Optional, Type, TypeVar, Literal, Union, Set, TYPE_CHECKING
def partial_log(self, limit: Optional[int]=None, fields: Optional[List[str]]=None, pretty: Optional[bool]=True, colored: Optional[bool]=False) -> str:
    """
        Returns the partial log as a string
        """
    s = f'[{self.__class__.__name__}]'
    if colored:
        s = f'|g|{s}|e|'
    if fields is None:
        fields = self.get_model_field_names()
    for field in fields:
        field_str = f'|g|{field}|e|' if colored else field
        val_s = f'\n\t{field_str}: {getattr(self, field)!r}' if pretty else f'{field_str}={getattr(self, field)!r}, '
        if limit is not None and len(val_s) > limit:
            val_s = f'{val_s[:limit]}...'
        s += val_s
    return s