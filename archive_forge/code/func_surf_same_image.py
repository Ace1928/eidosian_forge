import os
import unittest
import ctypes
import weakref
import gc
import pathlib
import platform
import pygame
def surf_same_image(a, b):
    """Return True if a's pixel buffer is identical to b's"""
    a_sz = a.get_height() * a.get_pitch()
    b_sz = b.get_height() * b.get_pitch()
    if a_sz != b_sz:
        return False
    a_bytes = ctypes.string_at(a._pixels_address, a_sz)
    b_bytes = ctypes.string_at(b._pixels_address, b_sz)
    return a_bytes == b_bytes