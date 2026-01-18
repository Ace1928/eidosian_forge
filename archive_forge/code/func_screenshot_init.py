from __future__ import annotations
import html
import typing
from urwid import str_util
from urwid.event_loop import ExitMainLoop
from urwid.util import get_encoding
from .common import AttrSpec, BaseScreen
def screenshot_init(sizes: list[tuple[int, int]], keys: list[list[str]]) -> None:
    """
    Replace curses_display.Screen and raw_display.Screen class with
    HtmlGenerator.

    Call this function before executing an application that uses
    curses_display.Screen to have that code use HtmlGenerator instead.

    sizes -- list of ( columns, rows ) tuples to be returned by each call
             to HtmlGenerator.get_cols_rows()
    keys -- list of lists of keys to be returned by each call to
            HtmlGenerator.get_input()

    Lists of keys may include "window resize" to force the application to
    call get_cols_rows and read a new screen size.

    For example, the following call will prepare an application to:
     1. start in 80x25 with its first call to get_cols_rows()
     2. take a screenshot when it calls draw_screen(..)
     3. simulate 5 "down" keys from get_input()
     4. take a screenshot when it calls draw_screen(..)
     5. simulate keys "a", "b", "c" and a "window resize"
     6. resize to 20x10 on its second call to get_cols_rows()
     7. take a screenshot when it calls draw_screen(..)
     8. simulate a "Q" keypress to quit the application

    screenshot_init( [ (80,25), (20,10) ],
        [ ["down"]*5, ["a","b","c","window resize"], ["Q"] ] )
    """
    for row, col in sizes:
        if not isinstance(row, int):
            raise TypeError(f'sizes must be list[tuple[int, int]], with values >0 : {row!r}')
        if row <= 0:
            raise ValueError(f'sizes must be list[tuple[int, int]], with values >0 : {row!r}')
        if not isinstance(col, int):
            raise TypeError(f'sizes must be list[tuple[int, int]], with values >0 : {col!r}')
        if col <= 0:
            raise ValueError(f'sizes must be list[tuple[int, int]], with values >0 : {col!r}')
    for line in keys:
        if not isinstance(line, list):
            raise TypeError(f'keys must be list[list[str]]: {line!r}')
        for k in line:
            if not isinstance(k, str):
                raise TypeError(f'keys must be list[list[str]]: {k!r}')
    from . import curses, raw
    curses.Screen = HtmlGenerator
    raw.Screen = HtmlGenerator
    HtmlGenerator.sizes = sizes
    HtmlGenerator.keys = keys