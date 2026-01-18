import sqlite3
import csv
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
def bulk_insert_data(conn, query, data):
    try:
        with conn.cursor() as cursor:
            cursor.executemany(query, data)
            conn.commit()
    except sqlite3.Error as e:
        print(f'Error during bulk insert: {e}')