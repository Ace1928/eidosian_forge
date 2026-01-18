from __future__ import annotations
import gettext
import importlib
import json
import locale
import os
import re
import sys
import traceback
from functools import lru_cache
from typing import Any, Pattern
import babel
from packaging.version import parse as parse_version
@staticmethod
def normalize_domain(domain: str) -> str:
    """Normalize a domain name.

        Parameters
        ----------
        domain: str
            Domain to normalize

        Returns
        -------
        str
            Normalized domain
        """
    return domain.replace('-', '_')