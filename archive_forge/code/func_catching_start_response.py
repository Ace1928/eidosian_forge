from __future__ import annotations
import os.path
import sys
import time
import typing as t
from pstats import Stats
def catching_start_response(status, headers, exc_info=None):
    start_response(status, headers, exc_info)
    return response_body.append