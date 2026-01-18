from __future__ import annotations
import argparse
import enum
import os
import typing as t
def get_comp_type() -> t.Optional[CompType]:
    """Parse the COMP_TYPE environment variable (if present) and return the associated CompType enum value."""
    value = os.environ.get('COMP_TYPE')
    comp_type = CompType(chr(int(value))) if value else None
    return comp_type