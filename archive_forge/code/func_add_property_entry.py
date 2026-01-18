import tkinter as tk
from tkinter import filedialog, messagebox
import json
import logging
from typing import Callable, Dict, List, Optional, Tuple, Any
def add_property_entry(self, panel: tk.Frame, property_name: str) -> None:
    """Helper to add entries to the properties panel with clear labeling and layout."""
    frame = tk.Frame(panel)
    frame.pack(fill=tk.X, padx=5, pady=5)
    tk.Label(frame, text=property_name.capitalize()).pack(side=tk.LEFT)
    entry = tk.Entry(frame)
    entry.pack(side=tk.RIGHT, fill=tk.X, expand=True)
    self.properties_entries[property_name] = entry