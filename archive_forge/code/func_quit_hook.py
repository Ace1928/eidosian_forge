import sys
import unittest
import platform
import pygame
def quit_hook():
    global quit_count
    quit_count += 1