import importlib
import os
import sys
import time
from ast import literal_eval
from datetime import datetime, timedelta, timezone
from enum import Enum
from functools import partial, update_wrapper
from json import JSONDecodeError, loads
from shutil import get_terminal_size
import click
from redis import Redis
from redis.sentinel import Sentinel
from rq.defaults import (
from rq.logutils import setup_loghandlers
from rq.utils import import_attribute, parse_timeout
from rq.worker import WorkerStatus
def parse_function_arg(argument, arg_pos):

    class ParsingMode(Enum):
        PLAIN_TEXT = 0
        JSON = 1
        LITERAL_EVAL = 2
    keyword = None
    if argument.startswith(':'):
        mode = ParsingMode.JSON
        value = argument[1:]
    elif argument.startswith('%'):
        mode = ParsingMode.LITERAL_EVAL
        value = argument[1:]
    else:
        index = argument.find('=')
        if index > 0:
            if ':' in argument and argument.index(':') + 1 == index:
                mode = ParsingMode.JSON
                keyword = argument[:index - 1]
            elif '%' in argument and argument.index('%') + 1 == index:
                mode = ParsingMode.LITERAL_EVAL
                keyword = argument[:index - 1]
            else:
                mode = ParsingMode.PLAIN_TEXT
                keyword = argument[:index]
            value = argument[index + 1:]
        else:
            mode = ParsingMode.PLAIN_TEXT
            value = argument
    if value.startswith('@'):
        try:
            with open(value[1:], 'r') as file:
                value = file.read()
        except FileNotFoundError:
            raise click.FileError(value[1:], 'Not found')
    if mode == ParsingMode.JSON:
        try:
            value = loads(value)
        except JSONDecodeError:
            raise click.BadParameter('Unable to parse %s as JSON.' % (keyword or '%s. non keyword argument' % arg_pos))
    elif mode == ParsingMode.LITERAL_EVAL:
        try:
            value = literal_eval(value)
        except Exception:
            raise click.BadParameter('Unable to eval %s as Python object. See https://docs.python.org/3/library/ast.html#ast.literal_eval' % (keyword or '%s. non keyword argument' % arg_pos))
    return (keyword, value)