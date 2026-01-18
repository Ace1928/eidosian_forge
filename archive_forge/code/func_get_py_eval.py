import re
from . import utilities
def get_py_eval(text):
    """
    From a ptolemy solutions file, extract the PY=EVAL=SECTION
    """
    return eval(utilities.join_long_lines(find_unique_section(text, 'PY=EVAL=SECTION')))