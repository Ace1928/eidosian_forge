import re
from hacking import core
import pycodestyle
@core.flake8ext
def mutable_default_arguments(physical_line, logical_line, filename):
    if pycodestyle.noqa(physical_line):
        return
    if mutable_default_argument_check.match(logical_line):
        yield (0, 'D701: Default parameter value is a mutable type')