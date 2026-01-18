from __future__ import annotations
import ipaddress
import re
from typing import Protocol, Sequence, Union, runtime_checkable
import attr
from .exceptions import (
@property
def pattern_class(self) -> type[CertificatePattern]:
    ...