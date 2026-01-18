import sqlite3
import csv
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
def close_database_resources(conn, cursor=None):
    if cursor:
        cursor.close()
    if conn:
        conn.close()