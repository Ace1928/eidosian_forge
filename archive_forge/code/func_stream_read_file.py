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
def stream_read_file(self, read_method):
    """
        Handle the reading of large files using a streaming approach to minimize memory consumption.

        :param read_method: The method to be used for reading the file.
        :return: The content of the file, appropriately formatted.
        """
    try:
        if read_method.__name__ in ['read_csv', 'read_text', 'read_json']:
            return read_method(chunksize=1024 * 1024 * 50)
        else:
            logging.warning(f'Streaming not implemented for {read_method.__name__}, using default read method.')
            return read_method()
    except Exception as e:
        logging.error(f'Error streaming file {self.file_path}: {str(e)}')
        raise