import os
import unittest
import ctypes
import weakref
import gc
import pathlib
import platform
import pygame
def test_freetype_Font_get_metrics(self):
    font = self._TEST_FONTS['sans']
    metrics = font.get_metrics('ABCD', size=24)
    self.assertEqual(len(metrics), len('ABCD'))
    self.assertIsInstance(metrics, list)
    for metrics_tuple in metrics:
        self.assertIsInstance(metrics_tuple, tuple, metrics_tuple)
        self.assertEqual(len(metrics_tuple), 6)
        for m in metrics_tuple[:4]:
            self.assertIsInstance(m, int)
        for m in metrics_tuple[4:]:
            self.assertIsInstance(m, float)
    metrics = font.get_metrics('', size=24)
    self.assertEqual(metrics, [])
    self.assertRaises(TypeError, font.get_metrics, 24, 24)
    self.assertRaises(RuntimeError, nullfont().get_metrics, 'a', size=24)