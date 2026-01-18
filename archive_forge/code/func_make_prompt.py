from typing import Any, Generic, List, Optional, TextIO, TypeVar, Union, overload
from . import get_console
from .console import Console
from .text import Text, TextType
def make_prompt(self, default: DefaultType) -> Text:
    """Make prompt text.

        Args:
            default (DefaultType): Default value.

        Returns:
            Text: Text to display in prompt.
        """
    prompt = self.prompt.copy()
    prompt.end = ''
    if self.show_choices and self.choices:
        _choices = '/'.join(self.choices)
        choices = f'[{_choices}]'
        prompt.append(' ')
        prompt.append(choices, 'prompt.choices')
    if default != ... and self.show_default and isinstance(default, (str, self.response_type)):
        prompt.append(' ')
        _default = self.render_default(default)
        prompt.append(_default)
    prompt.append(self.prompt_suffix)
    return prompt