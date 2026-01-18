import logging
import sys
import time
import uuid
import pytest
import panel as pn
def print_to_terminal(term):
    sys.stdout = term
    print('This print statement is redirected from stdout to the Panel Terminal')
    sys.stdout = sys.__stdout__
    print('This print statement is again redirected to the server console')