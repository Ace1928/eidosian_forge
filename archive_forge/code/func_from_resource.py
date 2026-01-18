import os
from collections import abc
from functools import partial
from gi.repository import GLib, GObject, Gio
@classmethod
def from_resource(cls, resource_path):
    return cls(resource_path=resource_path)