import os
import re
import string
import textwrap
import warnings
from string import Formatter
from pathlib import Path
from typing import List, Dict, Tuple, Optional, cast
def get_nlstr(self):
    try:
        return self.__nlstr
    except AttributeError:
        self.__nlstr = '\n'.join(self)
        return self.__nlstr