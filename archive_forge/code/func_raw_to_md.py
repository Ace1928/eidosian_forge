from __future__ import annotations
from .nbbase import nbformat, nbformat_minor
def raw_to_md(cell):
    """let raw passthrough as markdown"""
    cell.cell_type = 'markdown'