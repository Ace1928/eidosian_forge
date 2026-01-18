import re
import string
import copy
from ..core.token import Token
from .tokenizer import Tokenizer
from .tokenizer import TOKEN
from .options import BeautifierOptions
from ..core.output import Output
def remove_redundant_indentation(output, frame):
    if frame.multiline_frame or frame.mode == MODE.ForInitializer or frame.mode == MODE.Conditional:
        return
    output.remove_indent(frame.start_line_index)