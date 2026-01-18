import abc
import functools
import re
import tokenize
from tokenize import untokenize, TokenError
from io import StringIO
from IPython.core.splitinput import LineInfo
from IPython.utils import tokenutil
@CoroutineInputTransformer.wrap
def classic_prompt():
    """Strip the >>>/... prompts of the Python interactive shell."""
    prompt_re = re.compile('^(>>>|\\.\\.\\.)( |$)')
    initial_re = re.compile('^>>>( |$)')
    turnoff_re = re.compile('^[%!]')
    return _strip_prompts(prompt_re, initial_re, turnoff_re)