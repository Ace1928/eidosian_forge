import sqlite3
import csv
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
def save_messages(messages, output_format):
    filename = f'messages.{output_format}'
    if output_format == 'csv':
        save_as_csv(messages, filename)
    elif output_format == 'json':
        save_as_json(messages, filename)
    elif output_format == 'txt':
        save_as_txt(messages, filename)
    messagebox.showinfo('Info', f'Messages successfully extracted to {filename}.')