import os
from breezy import trace
from breezy.rename_map import RenameMap
from breezy.tests import TestCaseWithTransport
def my_note(fmt, *args):
    notes.append(fmt % args)