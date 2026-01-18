from . import processMagmaFile
from . import processRurFile
from . import processComponents
def parse_decomposition_from_file(filename):
    with open(filename, 'r') as f:
        return parse_decomposition(f.read())