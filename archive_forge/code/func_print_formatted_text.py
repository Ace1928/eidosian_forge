from __future__ import annotations
from asyncio.events import AbstractEventLoop
from typing import TYPE_CHECKING, Any, TextIO
from prompt_toolkit.application import Application
from prompt_toolkit.application.current import get_app_or_none, get_app_session
from prompt_toolkit.application.run_in_terminal import run_in_terminal
from prompt_toolkit.formatted_text import (
from prompt_toolkit.input import DummyInput
from prompt_toolkit.layout import Layout
from prompt_toolkit.output import ColorDepth, Output
from prompt_toolkit.output.defaults import create_output
from prompt_toolkit.renderer import (
from prompt_toolkit.styles import (
def print_formatted_text(*values: Any, sep: str=' ', end: str='\n', file: TextIO | None=None, flush: bool=False, style: BaseStyle | None=None, output: Output | None=None, color_depth: ColorDepth | None=None, style_transformation: StyleTransformation | None=None, include_default_pygments_style: bool=True) -> None:
    """
    ::

        print_formatted_text(*values, sep=' ', end='\\n', file=None, flush=False, style=None, output=None)

    Print text to stdout. This is supposed to be compatible with Python's print
    function, but supports printing of formatted text. You can pass a
    :class:`~prompt_toolkit.formatted_text.FormattedText`,
    :class:`~prompt_toolkit.formatted_text.HTML` or
    :class:`~prompt_toolkit.formatted_text.ANSI` object to print formatted
    text.

    * Print HTML as follows::

        print_formatted_text(HTML('<i>Some italic text</i> <ansired>This is red!</ansired>'))

        style = Style.from_dict({
            'hello': '#ff0066',
            'world': '#884444 italic',
        })
        print_formatted_text(HTML('<hello>Hello</hello> <world>world</world>!'), style=style)

    * Print a list of (style_str, text) tuples in the given style to the
      output.  E.g.::

        style = Style.from_dict({
            'hello': '#ff0066',
            'world': '#884444 italic',
        })
        fragments = FormattedText([
            ('class:hello', 'Hello'),
            ('class:world', 'World'),
        ])
        print_formatted_text(fragments, style=style)

    If you want to print a list of Pygments tokens, wrap it in
    :class:`~prompt_toolkit.formatted_text.PygmentsTokens` to do the
    conversion.

    If a prompt_toolkit `Application` is currently running, this will always
    print above the application or prompt (similar to `patch_stdout`). So,
    `print_formatted_text` will erase the current application, print the text,
    and render the application again.

    :param values: Any kind of printable object, or formatted string.
    :param sep: String inserted between values, default a space.
    :param end: String appended after the last value, default a newline.
    :param style: :class:`.Style` instance for the color scheme.
    :param include_default_pygments_style: `bool`. Include the default Pygments
        style when set to `True` (the default).
    """
    assert not (output and file)
    if output is None:
        if file:
            output = create_output(stdout=file)
        else:
            output = get_app_session().output
    assert isinstance(output, Output)
    color_depth = color_depth or output.get_default_color_depth()

    def to_text(val: Any) -> StyleAndTextTuples:
        if isinstance(val, list) and (not isinstance(val, FormattedText)):
            return to_formatted_text(f'{val}')
        return to_formatted_text(val, auto_convert=True)
    fragments = []
    for i, value in enumerate(values):
        fragments.extend(to_text(value))
        if sep and i != len(values) - 1:
            fragments.extend(to_text(sep))
    fragments.extend(to_text(end))

    def render() -> None:
        assert isinstance(output, Output)
        renderer_print_formatted_text(output, fragments, _create_merged_style(style, include_default_pygments_style=include_default_pygments_style), color_depth=color_depth, style_transformation=style_transformation)
        if flush:
            output.flush()
    loop: AbstractEventLoop | None = None
    app = get_app_or_none()
    if app is not None:
        loop = app.loop
    if loop is not None:
        loop.call_soon_threadsafe(lambda: run_in_terminal(render))
    else:
        render()