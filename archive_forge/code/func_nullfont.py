import os
import unittest
import ctypes
import weakref
import gc
import pathlib
import platform
import pygame
def nullfont():
    """return an uninitialized font instance"""
    return ft.Font.__new__(ft.Font)