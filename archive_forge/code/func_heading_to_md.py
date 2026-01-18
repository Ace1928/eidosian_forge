from __future__ import annotations
from .nbbase import nbformat, nbformat_minor
def heading_to_md(cell):
    """turn heading cell into corresponding markdown"""
    cell.cell_type = 'markdown'
    level = cell.pop('level', 1)
    cell.source = '#' * level + ' ' + cell.source