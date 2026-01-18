import pygame
def set_resolution(self, width, height):
    """Sets the capture resolution. (without dialog)"""
    self.dev.setresolution(width, height)