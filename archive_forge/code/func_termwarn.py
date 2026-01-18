import os
import sys
import tempfile
import time
import click
import wandb
from wandb import env
def termwarn(string, **kwargs):
    string = '\n'.join([f'{WARN_STRING} {s}' for s in string.split('\n')])
    termlog(string=string, newline=True, **kwargs)