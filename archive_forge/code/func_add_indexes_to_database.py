import sqlite3
import csv
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
def add_indexes_to_database(conn):
    index_query = 'CREATE INDEX IF NOT EXISTS idx_handle_id ON message(handle_id);'
    try:
        cursor = conn.cursor()
        cursor.execute(index_query)
        conn.commit()
    except sqlite3.Error as e:
        print(f'Error creating index: {e}')
    finally:
        cursor.close()