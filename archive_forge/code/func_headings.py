import abc
import datetime as dt
import textwrap
from osc_lib.command import command
@classmethod
def headings(cls):
    return [c[1] for c in cls.COLUMNS]