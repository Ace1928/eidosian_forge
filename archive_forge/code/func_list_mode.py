from __future__ import annotations
import argparse
import enum
import os
import typing as t
@property
def list_mode(self) -> bool:
    """True if completion is running in list mode, otherwise False."""
    return self in (CompType.LIST, CompType.LIST_AMBIGUOUS, CompType.LIST_UNMODIFIED)