import platform
import unittest
import pygame
from pygame.locals import *
from pygame.pixelcopy import surface_to_array, map_array, array_to_surface, make_surface
def test_map_array(self):
    try:
        from numpy import array, zeros, uint8, int32, alltrue
    except ImportError:
        return
    surf = pygame.Surface((1, 1), 0, 32)
    color = array([11, 17, 59], uint8)
    target = zeros((5, 7), int32)
    map_array(target, color, surf)
    self.assertTrue(alltrue(target == surf.map_rgb(color)))
    stripe = array([[2, 5, 7], [11, 19, 23], [37, 53, 101]], uint8)
    target = zeros((4, stripe.shape[0]), int32)
    map_array(target, stripe, surf)
    target_stripe = array([surf.map_rgb(c) for c in stripe], int32)
    self.assertTrue(alltrue(target == target_stripe))
    stripe = array([[[2, 5, 7]], [[11, 19, 24]], [[10, 20, 30]], [[37, 53, 101]]], uint8)
    target = zeros((stripe.shape[0], 3), int32)
    map_array(target, stripe, surf)
    target_stripe = array([[surf.map_rgb(c)] for c in stripe[:, 0]], int32)
    self.assertTrue(alltrue(target == target_stripe))
    w = 4
    h = 5
    source = zeros((w, h, 3), uint8)
    target = zeros((w,), int32)
    self.assertRaises(ValueError, map_array, target, source, surf)
    source = zeros((12, w, h + 1), uint8)
    self.assertRaises(ValueError, map_array, target, source, surf)
    source = zeros((12, w - 1, 5), uint8)
    self.assertRaises(ValueError, map_array, target, source, surf)