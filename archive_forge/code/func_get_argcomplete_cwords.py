from __future__ import annotations
import argparse
import os
import typing as t
def get_argcomplete_cwords() -> t.Optional[t.List[str]]:
    """Get current words prior to completion point

    This is normally done in the `argcomplete.CompletionFinder` constructor,
    but is exposed here to allow `traitlets` to follow dynamic code-paths such
    as determining whether to evaluate a subcommand.
    """
    if '_ARGCOMPLETE' not in os.environ:
        return None
    comp_line = os.environ['COMP_LINE']
    comp_point = int(os.environ['COMP_POINT'])
    comp_words: t.List[str]
    try:
        cword_prequote, cword_prefix, cword_suffix, comp_words, last_wordbreak_pos = argcomplete.split_line(comp_line, comp_point)
    except ModuleNotFoundError:
        return None
    start = int(os.environ['_ARGCOMPLETE']) - 1
    comp_words = comp_words[start:]
    return comp_words