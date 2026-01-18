from __future__ import unicode_literals
from .buffer import Buffer, AcceptAction
from .document import Document
from .enums import DEFAULT_BUFFER, SEARCH_BUFFER, EditingMode
from .filters import IsDone, HasFocus, RendererHeightIsKnown, to_simple_filter, to_cli_filter, Condition
from .history import InMemoryHistory
from .interface import CommandLineInterface, Application, AbortAction
from .key_binding.defaults import load_key_bindings_for_prompt
from .key_binding.registry import Registry
from .keys import Keys
from .layout import Window, HSplit, FloatContainer, Float
from .layout.containers import ConditionalContainer
from .layout.controls import BufferControl, TokenListControl
from .layout.dimension import LayoutDimension
from .layout.lexers import PygmentsLexer
from .layout.margins import PromptMargin, ConditionalMargin
from .layout.menus import CompletionsMenu, MultiColumnCompletionsMenu
from .layout.processors import PasswordProcessor, ConditionalProcessor, AppendAutoSuggestion, HighlightSearchProcessor, HighlightSelectionProcessor, DisplayMultipleCursors
from .layout.prompt import DefaultPrompt
from .layout.screen import Char
from .layout.toolbars import ValidationToolbar, SystemToolbar, ArgToolbar, SearchToolbar
from .layout.utils import explode_tokens
from .renderer import print_tokens as renderer_print_tokens
from .styles import DEFAULT_STYLE, Style, style_from_dict
from .token import Token
from .utils import is_conemu_ansi, is_windows, DummyContext
from six import text_type, exec_, PY2
import os
import sys
import textwrap
import threading
import time
def run_application(application, patch_stdout=False, return_asyncio_coroutine=False, true_color=False, refresh_interval=0, eventloop=None):
    """
    Run a prompt toolkit application.

    :param patch_stdout: Replace ``sys.stdout`` by a proxy that ensures that
            print statements from other threads won't destroy the prompt. (They
            will be printed above the prompt instead.)
    :param return_asyncio_coroutine: When True, return a asyncio coroutine. (Python >3.3)
    :param true_color: When True, use 24bit colors instead of 256 colors.
    :param refresh_interval: (number; in seconds) When given, refresh the UI
        every so many seconds.
    """
    assert isinstance(application, Application)
    if return_asyncio_coroutine:
        eventloop = create_asyncio_eventloop()
    else:
        eventloop = eventloop or create_eventloop()
    cli = CommandLineInterface(application=application, eventloop=eventloop, output=create_output(true_color=true_color))
    if refresh_interval:
        done = [False]

        def start_refresh_loop(cli):

            def run():
                while not done[0]:
                    time.sleep(refresh_interval)
                    cli.request_redraw()
            t = threading.Thread(target=run)
            t.daemon = True
            t.start()

        def stop_refresh_loop(cli):
            done[0] = True
        cli.on_start += start_refresh_loop
        cli.on_stop += stop_refresh_loop
    patch_context = cli.patch_stdout_context(raw=True) if patch_stdout else DummyContext()
    if return_asyncio_coroutine:
        exec_context = {'patch_context': patch_context, 'cli': cli, 'Document': Document}
        exec_(textwrap.dedent('\n        def prompt_coro():\n            # Inline import, because it slows down startup when asyncio is not\n            # needed.\n            import asyncio\n\n            @asyncio.coroutine\n            def run():\n                with patch_context:\n                    result = yield from cli.run_async()\n\n                if isinstance(result, Document):  # Backwards-compatibility.\n                    return result.text\n                return result\n            return run()\n        '), exec_context)
        return exec_context['prompt_coro']()
    else:
        try:
            with patch_context:
                result = cli.run()
            if isinstance(result, Document):
                return result.text
            return result
        finally:
            eventloop.close()