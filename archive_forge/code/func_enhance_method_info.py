import ast
import json
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import logging
from typing import List, Dict, Any, Optional, Union
import docstring_parser
def enhance_method_info(self, method_info, docstring):
    enhanced_doc = self.parse_docstring(docstring)
    method_info.update(enhanced_doc)