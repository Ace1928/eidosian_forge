import contextlib
import re
import textwrap
from typing import Iterable, List, Tuple, Union
def do_wrap_urls(text: str, url_idx: Iterable, text_idx: int, indentation: str, wrap_length: int) -> Tuple[List[str], int]:
    """Wrap URLs in the long description.

    Parameters
    ----------
    text : str
        The long description text.
    url_idx : list
        The list of URL indices found in the description text.
    text_idx : int
        The index in the description of the end of the last URL.
    indentation : str
        The string to use to indent each line in the long description.
    wrap_length : int
         The line length at which to wrap long lines in the description.

    Returns
    -------
    _lines, _text_idx : tuple
        A list of the long description lines and the index in the long
        description where the last URL ended.
    """
    _lines = []
    for _url in url_idx:
        if do_skip_link(text, _url):
            continue
        if len(text[text_idx:_url[1]]) > wrap_length - len(indentation):
            _lines.extend(description_to_list(text[text_idx:_url[0]], indentation, wrap_length))
            with contextlib.suppress(IndexError):
                if text[_url[0] - len(indentation) - 2] != '\n' and (not _lines[-1]):
                    _lines.pop(-1)
            _text = f'{text[_url[0]:_url[1]]}'
            with contextlib.suppress(IndexError):
                if _lines[0][-1] == '"':
                    _lines[0] = _lines[0][:-2]
                    _text = f'"{text[_url[0]:_url[1]]}'
            _lines.append(f'{do_clean_url(_text, indentation)}')
            text_idx = _url[1]
    return (_lines, text_idx)