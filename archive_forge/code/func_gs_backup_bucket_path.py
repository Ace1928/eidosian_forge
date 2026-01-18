import os
import json
import pathlib
from typing import Optional, Union, Dict, Any
from lazyops.types.models import BaseSettings, validator
from lazyops.types.classprops import lazyproperty
from lazyops.imports._fileio import (
@lazyproperty
@require_fileio()
def gs_backup_bucket_path(self):
    if self.gs_backup_bucket is None:
        return None
    bucket = self.gs_backup_bucket
    if not bucket.startswith('gs://'):
        bucket = f'gs://{bucket}'
    return File(bucket)