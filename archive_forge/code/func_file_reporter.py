import os
import coverage
from kivy.lang.parser import Parser
def file_reporter(self, filename):
    return KivyFileReporter(filename=filename)