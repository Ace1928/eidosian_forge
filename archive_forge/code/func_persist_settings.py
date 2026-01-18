import sys
import torch
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTextEdit, QPushButton, QVBoxLayout, QWidget,
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import os
import json
import logging
from typing import Dict, Tuple, Optional, Any
from datetime import datetime
from huggingface_hub import HfApi
def persist_settings(self) -> None:
    """Reliable settings persistence with detailed logging."""
    try:
        with open(self.settings_file, 'w') as file:
            json.dump(self.settings, file)
        log_verbose('Settings persisted successfully.')
    except IOError as e:
        log_verbose(f'Failed to persist settings: {e}')