import errno
import functools
import os
import re
import subprocess
import sys
from typing import Callable
@register_vcs_handler('git', 'keywords')
def git_versions_from_keywords(keywords, tag_prefix, verbose):
    """Get version information from git keywords."""
    if 'refnames' not in keywords:
        raise NotThisMethod('Short version file found')
    date = keywords.get('date')
    if date is not None:
        date = date.splitlines()[-1]
        date = date.strip().replace(' ', 'T', 1).replace(' ', '', 1)
    refnames = keywords['refnames'].strip()
    if refnames.startswith('$Format'):
        if verbose:
            print('keywords are unexpanded, not using')
        raise NotThisMethod('unexpanded keywords, not a git-archive tarball')
    refs = {r.strip() for r in refnames.strip('()').split(',')}
    TAG = 'tag: '
    tags = {r[len(TAG):] for r in refs if r.startswith(TAG)}
    if not tags:
        tags = {r for r in refs if re.search('\\d', r)}
        if verbose:
            print(f"discarding '{','.join(refs - tags)}', no digits")
    if verbose:
        print(f'likely tags: {','.join(sorted(tags))}')
    for ref in sorted(tags):
        if ref.startswith(tag_prefix):
            r = ref[len(tag_prefix):]
            if not re.match('\\d', r):
                continue
            if verbose:
                print(f'picking {r}')
            return {'version': r, 'full-revisionid': keywords['full'].strip(), 'dirty': False, 'error': None, 'date': date}
    if verbose:
        print('no suitable tags, using unknown + full revision id')
    return {'version': '0+unknown', 'full-revisionid': keywords['full'].strip(), 'dirty': False, 'error': 'no suitable tags', 'date': None}