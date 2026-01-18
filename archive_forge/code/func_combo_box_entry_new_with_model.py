import sys
import warnings
from collections import UserList
import gi
from gi.repository import GObject
def combo_box_entry_new_with_model(model):
    return Gtk.ComboBoxEntry(model=model)