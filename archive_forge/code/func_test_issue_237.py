import os
import unittest
import ctypes
import weakref
import gc
import pathlib
import platform
import pygame
def test_issue_237(self):
    """Issue #237: Memory overrun when rendered with underlining"""
    name = 'Times New Roman'
    font = ft.SysFont(name, 19)
    if font.name != name:
        return
    font.underline = True
    s, r = font.render('Amazon', size=19)
    for adj in [-2, -1.9, -1, 0, 1.9, 2]:
        font.underline_adjustment = adj
        s, r = font.render('Amazon', size=19)