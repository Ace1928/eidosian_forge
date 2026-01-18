from dataclasses import dataclass
from enum import IntEnum
from typing import Optional
from urllib.parse import urlparse
def job_reference(self) -> str:
    assert self.is_job()
    return f'{self.job_name}:{self.job_alias}'