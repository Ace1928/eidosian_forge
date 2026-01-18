from pyparsing import *
from sys import stdin, argv, exit
def setpos(self, location, text):
    """Helper function for setting curently parsed text and position"""
    self.location = location
    self.text = text