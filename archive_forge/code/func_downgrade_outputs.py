from __future__ import annotations
import json
import re
from traitlets.log import get_logger
from nbformat import v3, validator
from nbformat.corpus.words import generate_corpus_id as random_cell_id
from nbformat.notebooknode import NotebookNode
from .nbbase import nbformat, nbformat_minor
def downgrade_outputs(outputs):
    """downgrade outputs of a code cell to v3 from v4"""
    return [downgrade_output(op) for op in outputs]