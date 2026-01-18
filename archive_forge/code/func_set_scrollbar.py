import sys
import tkinter
from tkinter import ttk
from .zoom_slider import Slider
import time
def set_scrollbar(self, low, high):
    if float(low) <= 0.0 and float(high) >= 1.0:
        self.scrollbar.pack_forget()
        self.scrollbar.is_visible = False
    else:
        self.scrollbar.pack(side='right', fill='y', anchor='nw', pady=10)
        self.scrollbar.is_visible = True
    self.scrollbar.set(low, high)