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
def read_yaml(self) -> Dict[str, Any]:
    """
        Read a YAML file and return its content as a dictionary, ensuring that every possible element,
        including nested structures and metadata, is comprehensively extracted and represented in the
        returned dictionary. This method is meticulously crafted to handle various YAML structures with
        precision, providing a robust, complete, and perfect representation of the YAML content.

        :return: The content of the YAML file parsed into a dictionary, including all nested structures and metadata.
        :rtype: Dict[str, Any]
        """
    try:
        with open(self.file_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
            logging.debug(f'YAML data read successfully from {self.file_path}')
            logging.info(f'YAML data structure details: {json.dumps(data, indent=4)}')
            return data
    except yaml.YAMLError as e:
        logging.error(f'YAML parsing error at {self.file_path}: {str(e)}')
        raise Exception(f'YAML parsing error at {self.file_path}: {str(e)}')
    except Exception as e:
        logging.error(f'An error occurred while reading the YAML file at {self.file_path}: {str(e)}')
        raise Exception(f'An error occurred while reading the YAML file: {str(e)}')