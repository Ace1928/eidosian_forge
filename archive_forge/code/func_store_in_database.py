import json
import logging
import sqlite3
from typing import List, Any, Dict, Tuple
def store_in_database(vector_data: List[Any], db_path: str):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('CREATE TABLE IF NOT EXISTS vector_data (id INTEGER PRIMARY KEY, data TEXT)')
        for entry in vector_data:
            cursor.execute('INSERT INTO vector_data (data) VALUES (?)', (str(entry),))
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        log_error(f'SQLite error: {e}')