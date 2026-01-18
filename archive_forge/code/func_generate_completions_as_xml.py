from collections import namedtuple
from string import ascii_letters, digits
from _pydevd_bundle import pydevd_xml
import pydevconsole
import builtins as __builtin__  # Py3
def generate_completions_as_xml(frame, act_tok):
    completions = generate_completions(frame, act_tok)
    return completions_to_xml(completions)