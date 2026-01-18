import os
import json
import csv
import xml.etree.ElementTree as ET
from typing import Any, Dict, Union, Optional, Tuple, List
import pandas as pd
import logging
import yaml
import pickle
import configparser
import markdown
import openpyxl
import sqlite3
import PyPDF2
import PIL.Image
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, Label, Toplevel
from PIL import Image, ImageTk
import os
import logging
import json
import pandas as pd
def display_content(self, content):
    if isinstance(content, pd.DataFrame):
        self.show_dataframe(content)
    elif isinstance(content, (Image.Image, ImageTk.PhotoImage)):
        self.show_image(content)
    elif isinstance(content, dict):
        formatted_content = json.dumps(content, indent=4)
        self.show_text(formatted_content)
    elif isinstance(content, list):
        formatted_content = '\n'.join((str(item) for item in content))
        self.show_text(formatted_content)
    else:
        self.show_text(str(content))