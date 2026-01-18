from __future__ import annotations
import contextlib
from lazyops.types import BaseModel, Field, root_validator
from lazyops.types.models import ConfigDict, schema_extra
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException
from kvdb.types.jobs import Job, JobStatus
from lazyops.libs.logging import logger
from typing import Any, Dict, List, Optional, Type, TypeVar, Literal, Union, Set, TYPE_CHECKING
def get_excluded_function_params(self, key: Optional[str]='exclude_param') -> List[str]:
    """
        Returns the excluded param fields
        """
    return [field_name for field_name, field in self.model_fields.items() if field.json_schema_extra.get(key, False)]