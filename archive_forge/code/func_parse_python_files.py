import ast
import json
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import logging
from typing import List, Dict, Any, Optional, Union
import docstring_parser
def parse_python_files(filepaths: List[str]) -> List[Dict[str, Any]]:
    classes: List[Dict[str, Any]] = []
    for filepath in filepaths:
        logging.info(f'Attempting to parse file: {filepath}')
        try:
            with open(filepath, 'r') as file:
                node = ast.parse(file.read(), filename=os.path.basename(filepath))
                parser = CodeParser()
                parser.visit(node)
                classes.extend(parser.classes)
                logging.info(f'Successfully parsed {filepath}.')
        except SyntaxError as e:
            messagebox.showerror('Error', f'Syntax error in {os.path.basename(filepath)}: {e}')
            logging.error(f'Syntax error in {filepath}: {e}')
        except Exception as e:
            messagebox.showerror('Error', f'Failed to parse {os.path.basename(filepath)}: {e}')
            logging.error(f'Failed to parse {filepath}: {e}')
    return classes