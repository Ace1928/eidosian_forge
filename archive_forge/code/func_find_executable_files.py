import os
import coverage
from kivy.lang.parser import Parser
def find_executable_files(self, src_dir):
    for dirpath, dirnames, filenames in os.walk(src_dir):
        for filename in filenames:
            if filename.endswith('.kv'):
                yield os.path.join(dirpath, filename)