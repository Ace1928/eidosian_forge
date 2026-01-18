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
def launch_model_browser(self) -> None:
    """Initiates the model browsing dialog, enabling users to select different AI models."""
    browser = ModelBrowser(self)
    if browser.exec_():
        new_model_id = browser.selected_model_id
        if new_model_id:
            self.settings_manager.update_setting('model_id', new_model_id)
            self.load_model_and_tokenizer(new_model_id)
            log_verbose(f'Model updated: {new_model_id}')