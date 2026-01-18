from __future__ import annotations
import argparse
import os
import typing as t
def match_class_completions(self, cword_prefix: str) -> t.List[t.Tuple[t.Any, str]]:
    """Match the word to be completed against our Configurable classes

        Check if cword_prefix could potentially match against --{class}. for any class
        in Application.classes.
        """
    class_completions = [(cls, f'--{cls.__name__}.') for cls in self.config_classes]
    matched_completions = class_completions
    if '.' in cword_prefix:
        cword_prefix = cword_prefix[:cword_prefix.index('.') + 1]
        matched_completions = [(cls, c) for cls, c in class_completions if c == cword_prefix]
    elif len(cword_prefix) > 0:
        matched_completions = [(cls, c) for cls, c in class_completions if c.startswith(cword_prefix)]
    return matched_completions