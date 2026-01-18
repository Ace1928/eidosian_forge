from __future__ import annotations
import abc
import collections.abc as c
import contextlib
import dataclasses
import enum
import os
import re
import typing as t
@property
def current_boundary(self) -> t.Optional[ParserBoundary]:
    """The current parser boundary, if any, otherwise None."""
    return self.boundaries[-1] if self.boundaries else None