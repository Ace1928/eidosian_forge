import ast
import json
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import logging
from typing import List, Dict, Any, Optional, Union
import docstring_parser
def extract_return_info(self, node: ast.FunctionDef) -> Dict[str, str]:
    return_annotation = node.returns
    return_type: str = self.infer_type(return_annotation) if return_annotation else 'Return type not specified'
    return {'Type': return_type, 'Description': 'Detailed return information not available.'}