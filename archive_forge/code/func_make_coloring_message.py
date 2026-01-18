import re
from string import Formatter
def make_coloring_message(self, message, *, ansi_level, colored_message):
    messages = [message if color_tokens is None else AnsiParser.wrap(colored_message.tokens, ansi_level=ansi_level, color_tokens=color_tokens) for color_tokens in self._messages_color_tokens]
    coloring = ColoringMessage(message)
    coloring._messages = iter(messages)
    return coloring