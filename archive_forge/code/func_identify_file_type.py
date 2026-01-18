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
def identify_file_type(self) -> str:
    """
        Determine the file type by extracting and analyzing the file extension.

        :return: A string representing the file type.
        :rtype: str
        """
    _, file_extension = os.path.splitext(self.file_path)
    file_extension = file_extension.lower()
    logging.debug(f'File extension identified: {file_extension}')
    return file_extension