from __future__ import annotations
import gzip
import io
import typing as ty
from bz2 import BZ2File
from os.path import splitext
from ._compression import HAVE_INDEXED_GZIP, IndexedGzipFile, pyzstd
def close_if_mine(self) -> None:
    """Close ``self.fobj`` iff we opened it in the constructor"""
    if self.me_opened:
        self.close()