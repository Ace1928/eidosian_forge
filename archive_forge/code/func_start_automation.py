import pyautogui
import time
import datetime
import os
import json
from tkinter import Tk, Label, Button, Entry, StringVar
import numpy as np
import cv2
import pyperclip
def start_automation(self):
    """
        Start the automation process based on the user-defined settings.
        """
    settings = {'chat_window_1': {'coords': (int(self.entry_chat1_x.get()), int(self.entry_chat1_y.get()))}, 'chat_window_2': {'coords': (int(self.entry_chat2_x.get()), int(self.entry_chat2_y.get()))}, 'capture_area': (int(self.entry_area_width.get()), int(self.entry_area_height.get()))}
    self.automated_conversation(settings)