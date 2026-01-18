import pyautogui
import time
import datetime
import os
import json
from tkinter import Tk, Label, Button, Entry, StringVar
import numpy as np
import cv2
import pyperclip
def setup_buttons(self):
    """
        Setup buttons for starting the automation and saving settings.
        """
    self.start_button = Button(self.root, text='Start Automation', command=self.start_automation)
    self.save_button = Button(self.root, text='Save Settings', command=self.save_settings)
    self.start_button.grid(row=4, column=1)
    self.save_button.grid(row=4, column=2)