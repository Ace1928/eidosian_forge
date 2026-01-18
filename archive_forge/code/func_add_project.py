import io
import sys
from typing import Dict, Any, Set
from pathlib import Path
from flask import Flask, render_template, request
from ase.db import connect
from ase.db.core import Database
from ase.formula import Formula
from ase.db.web import create_key_descriptions, Session
from ase.db.row import row2dct, AtomsRow
from ase.db.table import all_columns
def add_project(db: Database) -> None:
    """Add database to projects with name 'default'."""
    all_keys: Set[str] = set()
    for row in db.select(columns=['key_value_pairs'], include_data=False):
        all_keys.update(row._keys)
    key_descriptions = {key: (key, '', '') for key in all_keys}
    meta: Dict[str, Any] = db.metadata
    if 'key_descriptions' in meta:
        key_descriptions.update(meta['key_descriptions'])
    default_columns = meta.get('default_columns')
    if default_columns is None:
        default_columns = all_columns[:]
    projects['default'] = {'name': 'default', 'title': meta.get('title', ''), 'uid_key': 'id', 'key_descriptions': create_key_descriptions(key_descriptions), 'database': db, 'row_to_dict_function': row_to_dict, 'handle_query_function': handle_query, 'default_columns': default_columns, 'search_template': 'ase/db/templates/search.html', 'row_template': 'ase/db/templates/row.html', 'table_template': 'ase/db/templates/table.html'}