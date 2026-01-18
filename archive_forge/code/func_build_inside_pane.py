import os
import sys
from .gui import *
from .app_menus import ListedWindow
def build_inside_pane(self, master):
    groupBG = self.style.groupBG
    self.keyboard = Tk_.Variable(value=self.settings['keyboard'])
    self.inside_frame = frame = ttk.Frame(master)
    frame.rowconfigure(3, weight=1)
    frame.columnconfigure(0, weight=1)
    frame.columnconfigure(3, weight=1)
    strut = ttk.Frame(frame, width=1)
    strut.grid(rowspan=5, column=0)
    keyboard_label = ttk.Label(frame, anchor=Tk_.W, text='Which keyboard layout are you using?')
    keyboard_label.grid(row=0, column=1, columnspan=2, sticky=Tk_.W, pady=(20, 0))
    keyboard_button = ttk.OptionMenu(frame, self.keyboard, self.keyboard.get(), 'QWERTY', 'AZERTY', 'QWERTZ', command=self.set_keyboard)
    keyboard_button.grid(row=1, column=1, columnspan=2, sticky=Tk_.W, pady=(10, 0))