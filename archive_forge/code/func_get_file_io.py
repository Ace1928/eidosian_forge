import os
from parso import file_io
def get_file_io(self, name):
    return FileIO(os.path.join(self.path, name))