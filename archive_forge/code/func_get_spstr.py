import os
import re
import string
import textwrap
import warnings
from string import Formatter
from pathlib import Path
from typing import List, Dict, Tuple, Optional, cast
def get_spstr(self):
    try:
        return self.__spstr
    except AttributeError:
        self.__spstr = ' '.join(self)
        return self.__spstr