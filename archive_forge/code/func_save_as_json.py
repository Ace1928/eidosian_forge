import sqlite3
import csv
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
def save_as_json(messages, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump([{'text': m[0], 'date': m[1], 'is_from_me': m[2]} for m in messages], file, ensure_ascii=False, indent=4)