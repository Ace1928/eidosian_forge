import tkinter as tk
from tkinter import filedialog, messagebox
import json
import logging
from typing import Callable, Dict, List, Optional, Tuple, Any
def add_menu(self, menu_bar: tk.Menu, label: str, commands: List[Tuple[str, Callable]]) -> None:
    """Helper to add dropdown menus to the menu bar with explicit command linking."""
    menu = tk.Menu(menu_bar, tearoff=0)
    for command in commands:
        if command is None:
            menu.add_separator()
        else:
            menu.add_command(label=command[0], command=command[1])
    menu_bar.add_cascade(label=label, menu=menu)