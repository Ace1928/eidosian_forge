import os
import pygame as pg
def punched(self):
    """this will cause the monkey to start spinning"""
    if not self.dizzy:
        self.dizzy = True
        self.original = self.image