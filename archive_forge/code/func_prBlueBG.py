import os
import json
import random
import tempfile
import subprocess
from parlai.core.params import ParlaiParser
from parlai.core.script import ParlaiScript, register_script
def prBlueBG(text):
    """
    Print given in text with a blue background.

    :param text: The text to be printed
    """
    print('\x1b[44m{}\x1b[0m'.format(text), sep='')