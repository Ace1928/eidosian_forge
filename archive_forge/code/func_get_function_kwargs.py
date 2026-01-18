from __future__ import annotations
import contextlib
from lazyops.types import BaseModel, Field, root_validator
from lazyops.types.models import ConfigDict, schema_extra
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException
from kvdb.types.jobs import Job, JobStatus
from lazyops.libs.logging import logger
from typing import Any, Dict, List, Optional, Type, TypeVar, Literal, Union, Set, TYPE_CHECKING
def get_function_kwargs(self, exclude_unset: Optional[bool]=True, exclude: Optional[Set[str]]=None, **kwargs) -> Dict[str, Any]:
    """
        Returns the Function Kwargs
        """
    exclude = exclude or set()
    excluded_params = self.get_excluded_function_params()
    if excluded_params:
        exclude.update(excluded_params)
    return self.model_dump(exclude_unset=exclude_unset, exclude=exclude, **kwargs)