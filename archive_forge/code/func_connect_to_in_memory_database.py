import sqlite3
import csv
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
def connect_to_in_memory_database():
    try:
        conn = sqlite3.connect(':memory:')
        return conn
    except sqlite3.Error as e:
        print(f'Error connecting to in-memory database: {e}')