import os
import pygame
import sys
import tempfile
import time
def rgba_between(value, minimum=0, maximum=255):
    if value < minimum:
        return minimum
    elif value > maximum:
        return maximum
    else:
        return value