from __future__ import annotations
import datetime
from typing import Any, Dict, List, Optional, Union
def remove_sql_comments(sql_text: str) -> str:
    """
    Removes the SQL comments from the given SQL text
    """
    while '/*' in sql_text:
        sql_text = sql_text[:sql_text.index('/*')] + sql_text[sql_text.index('*/') + 2:].strip()
    return sql_text