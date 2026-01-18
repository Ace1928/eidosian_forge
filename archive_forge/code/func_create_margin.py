from __future__ import unicode_literals
from abc import ABCMeta, abstractmethod
from six import with_metaclass
from six.moves import range
from prompt_toolkit.filters import to_cli_filter
from prompt_toolkit.token import Token
from prompt_toolkit.utils import get_cwidth
from .utils import token_list_to_text
def create_margin(self, cli, window_render_info, width, height):
    tokens = self.get_prompt_tokens(cli)[:]
    if self.get_continuation_tokens:
        tokens2 = list(self.get_continuation_tokens(cli, width))
    else:
        tokens2 = []
    show_numbers = self.show_numbers(cli)
    last_y = None
    for y in window_render_info.displayed_lines[1:]:
        tokens.append((Token, '\n'))
        if show_numbers:
            if y != last_y:
                tokens.append((Token.LineNumber, ('%i ' % (y + 1)).rjust(width)))
        else:
            tokens.extend(tokens2)
        last_y = y
    return tokens