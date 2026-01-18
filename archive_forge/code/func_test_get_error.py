import os
import unittest
import ctypes
import weakref
import gc
import pathlib
import platform
import pygame
def test_get_error(self):
    """Ensures get_error() is initially empty (None)."""
    error_msg = ft.get_error()
    self.assertIsNone(error_msg)