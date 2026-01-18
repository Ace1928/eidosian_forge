import contextlib
import errno
import itertools
import logging
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import unicodedata
from enum import Enum
from types import FrameType, TracebackType
from typing import (
from .._typing_compat import Literal
import greenlet
from curtsies import (
from curtsies.configfile_keynames import keymap as key_dispatch
from curtsies.input import is_main_thread
from curtsies.window import CursorAwareWindow
from cwcwidth import wcswidth
from pygments import format as pygformat
from pygments.formatters import TerminalFormatter
from pygments.lexers import Python3Lexer
from . import events as bpythonevents, sitefix, replpainter as paint
from ..config import Config
from .coderunner import (
from .filewatch import ModuleChangedEventHandler
from .interaction import StatusBar
from .interpreter import (
from .manual_readline import (
from .parse import parse as bpythonparse, func_for_letter, color_for_letter
from .preprocess import preprocess
from .. import __version__
from ..config import getpreferredencoding
from ..formatter import BPythonFormatter
from ..pager import get_pager_command
from ..repl import (
from ..translations import _
from ..line import CHARACTER_PAIR_MAP
def paint(self, about_to_exit=False, user_quit=False, try_preserve_history_height=30, min_infobox_height=5) -> Tuple[FSArray, Tuple[int, int]]:
    """Returns an array of min_height or more rows and width columns, plus
        cursor position

        Paints the entire screen - ideally the terminal display layer will take
        a diff and only write to the screen in portions that have changed, but
        the idea is that we don't need to worry about that here, instead every
        frame is completely redrawn because less state is cool!

        try_preserve_history_height is the the number of rows of content that
        must be visible before the suggestion box scrolls the terminal in order
        to display more than min_infobox_height rows of suggestions, docs etc.
        """
    if about_to_exit:
        self.clean_up_current_line_for_exit()
    width, min_height = (self.width, self.height)
    show_status_bar = (bool(self.status_bar.should_show_message) or self.status_bar.has_focus) and (not self.request_paint_to_pad_bottom)
    if show_status_bar:
        min_height -= 1
    current_line_start_row = len(self.lines_for_display) - max(0, self.scroll_offset)
    if self.request_paint_to_clear_screen:
        self.request_paint_to_clear_screen = False
        arr = FSArray(min_height + current_line_start_row, width)
    elif self.request_paint_to_pad_bottom:
        height = min(self.request_paint_to_pad_bottom, min_height - 1)
        arr = FSArray(height, width)
        self.request_paint_to_pad_bottom = 0
    else:
        arr = FSArray(0, width)
    current_line = paint.paint_current_line(min_height, width, self.current_cursor_line)

    def move_screen_up(current_line_start_row):
        while current_line_start_row < 0:
            logger.debug('scroll_offset was %s, current_line_start_row was %s', self.scroll_offset, current_line_start_row)
            self.scroll_offset = self.scroll_offset - self.height
            current_line_start_row = len(self.lines_for_display) - max(-1, self.scroll_offset)
            logger.debug('scroll_offset changed to %s, current_line_start_row changed to %s', self.scroll_offset, current_line_start_row)
        return current_line_start_row
    if self.inconsistent_history and (not self.history_already_messed_up):
        logger.debug(INCONSISTENT_HISTORY_MSG)
        self.history_already_messed_up = True
        msg = INCONSISTENT_HISTORY_MSG
        arr[0, 0:min(len(msg), width)] = [msg[:width]]
        current_line_start_row += 1
        self.scroll_offset -= 1
        current_line_start_row = move_screen_up(current_line_start_row)
        logger.debug('current_line_start_row: %r', current_line_start_row)
        history = paint.paint_history(max(0, current_line_start_row - 1), width, self.lines_for_display)
        arr[1:history.height + 1, :history.width] = history
        if arr.height <= min_height:
            arr[min_height, 0] = ' '
    elif current_line_start_row < 0:
        logger.debug(CONTIGUITY_BROKEN_MSG)
        msg = CONTIGUITY_BROKEN_MSG
        arr[0, 0:min(len(msg), width)] = [msg[:width]]
        current_line_start_row = move_screen_up(current_line_start_row)
        history = paint.paint_history(max(0, current_line_start_row - 1), width, self.lines_for_display)
        arr[1:history.height + 1, :history.width] = history
        if arr.height <= min_height:
            arr[min_height, 0] = ' '
    else:
        assert current_line_start_row >= 0
        logger.debug('no history issues. start %i', current_line_start_row)
        history = paint.paint_history(current_line_start_row, width, self.lines_for_display)
        arr[:history.height, :history.width] = history
    self.inconsistent_history = False
    if user_quit:
        current_line_start_row = current_line_start_row - current_line.height
    logger.debug('---current line row slice %r, %r', current_line_start_row, current_line_start_row + current_line.height)
    logger.debug('---current line col slice %r, %r', 0, current_line.width)
    arr[current_line_start_row:current_line_start_row + current_line.height, 0:current_line.width] = current_line
    if current_line.height > min_height:
        return (arr, (0, 0))
    lines = paint.display_linize(self.current_cursor_line + 'X', width)
    current_line_end_row = current_line_start_row + len(lines) - 1
    current_line_height = current_line_end_row - current_line_start_row
    if self.stdin.has_focus:
        logger.debug('stdouterr when self.stdin has focus: %r %r', type(self.current_stdouterr_line), self.current_stdouterr_line)
        stdouterr = self.current_stdouterr_line
        if isinstance(stdouterr, FmtStr):
            stdouterr_width = stdouterr.width
        else:
            stdouterr_width = len(stdouterr)
        cursor_row, cursor_column = divmod(stdouterr_width + wcswidth(self.stdin.current_line, max(0, self.stdin.cursor_offset)), width)
        assert cursor_row >= 0 and cursor_column >= 0, (cursor_row, cursor_column, self.current_stdouterr_line, self.stdin.current_line)
    elif self.coderunner.running:
        cursor_row, cursor_column = divmod(len(self.current_cursor_line_without_suggestion) + self.cursor_offset, width)
        assert cursor_row >= 0 and cursor_column >= 0, (cursor_row, cursor_column, len(self.current_cursor_line), len(self.current_line), self.cursor_offset)
    else:
        cursor_row, cursor_column = divmod(wcswidth(self.current_cursor_line_without_suggestion.s) - wcswidth(self.current_line) + wcswidth(self.current_line, max(0, self.cursor_offset)) + self.number_of_padding_chars_on_current_cursor_line(), width)
        assert cursor_row >= 0 and cursor_column >= 0, (cursor_row, cursor_column, self.current_cursor_line_without_suggestion.s, self.current_line, self.cursor_offset)
    cursor_row += current_line_start_row
    if self.list_win_visible and (not self.coderunner.running):
        logger.debug('infobox display code running')
        visible_space_above = history.height
        potential_space_below = min_height - current_line_end_row - 1
        visible_space_below = potential_space_below - self.get_top_usable_line()
        if self.config.curtsies_list_above:
            info_max_rows = max(visible_space_above, visible_space_below)
        else:
            preferred_height = max(min_infobox_height, min_height - try_preserve_history_height)
            info_max_rows = min(max(visible_space_below, preferred_height), min_height - current_line_height - 1)
        infobox = paint.paint_infobox(info_max_rows, int(width * self.config.cli_suggestion_width), self.matches_iter.matches, self.funcprops, self.arg_pos, self.current_match, self.docstring, self.config, self.matches_iter.completer.format if self.matches_iter.completer else None)
        if visible_space_below >= infobox.height or not self.config.curtsies_list_above:
            arr[current_line_end_row + 1:current_line_end_row + 1 + infobox.height, 0:infobox.width] = infobox
        else:
            arr[current_line_start_row - infobox.height:current_line_start_row, 0:infobox.width] = infobox
            logger.debug('infobox of shape %r added to arr of shape %r', infobox.shape, arr.shape)
    logger.debug('about to exit: %r', about_to_exit)
    if show_status_bar:
        statusbar_row = min_height if arr.height == min_height else arr.height
        if about_to_exit:
            arr[statusbar_row, :] = FSArray(1, width)
        else:
            arr[statusbar_row, :] = paint.paint_statusbar(1, width, self.status_bar.current_line, self.config)
    if self.presentation_mode:
        rows = arr.height
        columns = arr.width
        last_key_box = paint.paint_last_events(rows, columns, [events.pp_event(x) for x in self.last_events if x], self.config)
        arr[arr.height - last_key_box.height:arr.height, arr.width - last_key_box.width:arr.width] = last_key_box
    if self.config.color_scheme['background'] not in ('d', 'D'):
        for r in range(arr.height):
            bg = color_for_letter(self.config.color_scheme['background'])
            arr[r] = fmtstr(arr[r], bg=bg)
    logger.debug('returning arr of size %r', arr.shape)
    logger.debug('cursor pos: %r', (cursor_row, cursor_column))
    return (arr, (cursor_row, cursor_column))