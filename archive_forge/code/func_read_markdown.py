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
def read_markdown(self) -> List[str]:
    """
        Read a Markdown file and return its content as a list of strings, each representing a line.
        This method ensures that every line of the Markdown file is read with precision and completeness,
        preserving the integrity and authenticity of the data.

        :return: The content of the Markdown file, with each line retained in its original form.
        :rtype: List[str]
        """
    try:
        with open(self.file_path, 'r', encoding='utf-8') as file:
            data = file.readlines()
            logging.debug(f'Markdown data read successfully from {self.file_path}')
            return data
    except FileNotFoundError:
        logging.error(f'Markdown file not found at {self.file_path}')
        raise FileNotFoundError(f'Markdown file not found at {self.file_path}')
    except Exception as e:
        logging.error(f'An error occurred while reading the Markdown file at {self.file_path}: {str(e)}')
        raise Exception(f'An error occurred while reading the Markdown file: {str(e)}')