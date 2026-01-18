import os
import unittest
import ctypes
import weakref
import gc
import pathlib
import platform
import pygame
def test_issue_565(self):
    """get_metrics supporting rotation/styles/size"""
    tests = [{'method': 'size', 'value': 36, 'msg': 'metrics same for size'}, {'method': 'rotation', 'value': 90, 'msg': 'metrics same for rotation'}, {'method': 'oblique', 'value': True, 'msg': 'metrics same for oblique'}]
    text = '|'

    def run_test(method, value, msg):
        font = ft.Font(self._sans_path, size=24)
        before = font.get_metrics(text)
        font.__setattr__(method, value)
        after = font.get_metrics(text)
        self.assertNotEqual(before, after, msg)
    for test in tests:
        run_test(test['method'], test['value'], test['msg'])