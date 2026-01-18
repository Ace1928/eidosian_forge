import os
import unittest
import ctypes
import weakref
import gc
import pathlib
import platform
import pygame
def ref_items(seq):
    return [weakref.ref(o) for o in seq]