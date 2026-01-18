import sys
import os
import time
import locale
import signal
import urwid
from typing import Optional
from . import args as bpargs, repl, translations
from .formatter import theme_map
from .translations import _
from .keys import urwid_key_dispatch as key_dispatch
def load_urwid_command_map(config):
    urwid.command_map[key_dispatch[config.up_one_line_key]] = 'cursor up'
    urwid.command_map[key_dispatch[config.down_one_line_key]] = 'cursor down'
    urwid.command_map[key_dispatch['C-a']] = 'cursor max left'
    urwid.command_map[key_dispatch['C-e']] = 'cursor max right'
    urwid.command_map[key_dispatch[config.pastebin_key]] = 'pastebin'
    urwid.command_map[key_dispatch['C-f']] = 'cursor right'
    urwid.command_map[key_dispatch['C-b']] = 'cursor left'
    urwid.command_map[key_dispatch['C-d']] = 'delete'
    urwid.command_map[key_dispatch[config.clear_word_key]] = 'clear word'
    urwid.command_map[key_dispatch[config.clear_line_key]] = 'clear line'