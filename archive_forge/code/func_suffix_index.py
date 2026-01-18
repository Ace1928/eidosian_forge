from __future__ import annotations
import logging
import os
import urllib.parse
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from functools import wraps
import idna
import requests
from .cache import DiskCache, get_cache_dir
from .remote import lenient_netloc, looks_like_ip, looks_like_ipv6
from .suffix_list import get_suffix_lists
def suffix_index(self, spl: list[str], include_psl_private_domains: bool | None=None) -> tuple[int, bool]:
    """Return the index of the first suffix label, and whether it is private.

        Returns len(spl) if no suffix is found.
        """
    if include_psl_private_domains is None:
        include_psl_private_domains = self.include_psl_private_domains
    node = self.tlds_incl_private_trie if include_psl_private_domains else self.tlds_excl_private_trie
    i = len(spl)
    j = i
    for label in reversed(spl):
        decoded_label = _decode_punycode(label)
        if decoded_label in node.matches:
            j -= 1
            node = node.matches[decoded_label]
            if node.end:
                i = j
            continue
        is_wildcard = '*' in node.matches
        if is_wildcard:
            is_wildcard_exception = '!' + decoded_label in node.matches
            if is_wildcard_exception:
                return (j, node.matches['*'].is_private)
            return (j - 1, node.matches['*'].is_private)
        break
    return (i, node.is_private)