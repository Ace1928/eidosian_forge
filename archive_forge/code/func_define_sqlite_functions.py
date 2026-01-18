import sqlite3
import csv
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
def define_sqlite_functions(conn):

    def encrypt_password(password):
        return encrypted_password
    conn.create_function('encrypt', 1, encrypt_password)