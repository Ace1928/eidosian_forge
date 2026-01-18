import collections.abc
import datetime
import email.utils
import functools
import logging
import io
import re
import subprocess
import warnings
import chardet
from debian._util import (
from debian.deprecation import function_deprecated_by
import debian.debian_support
import debian.changelog
@classmethod
def parse_relations(cls, raw):
    """Parse a package relationship string (i.e. the value of a field like
        Depends, Recommends, Build-Depends ...)
        """

    def parse_archs(raw):
        archs = []
        for arch in cls.__blank_sep_RE.split(raw.strip()):
            disabled = arch[0] == '!'
            if disabled:
                arch = arch[1:]
            archs.append(cls.ArchRestriction(not disabled, arch))
        return archs

    def parse_restrictions(raw):
        """ split a restriction formula into a list of restriction lists

            Each term in the restriction list is a namedtuple of form:

                (enabled, label)

            where
                enabled: bool: whether the restriction is positive or negative
                profile: the profile name of the term e.g. 'stage1'
            """
        restrictions = []
        groups = cls.__restriction_sep_RE.split(raw.lower().strip('<> '))
        for rgrp in groups:
            group = []
            for restriction in cls.__blank_sep_RE.split(rgrp):
                match = cls.__restriction_RE.match(restriction)
                if match:
                    parts = match.groupdict()
                    group.append(cls.BuildRestriction(parts['enabled'] != '!', parts['profile']))
            restrictions.append(group)
        return restrictions

    def parse_rel(raw):
        match = cls.__dep_RE.match(raw)
        if match:
            parts = match.groupdict()
            d = {'name': parts['name'], 'archqual': parts['archqual'], 'version': None, 'arch': None, 'restrictions': None}
            if parts['relop'] or parts['version']:
                d['version'] = (parts['relop'], parts['version'])
            if parts['archs']:
                d['arch'] = parse_archs(parts['archs'])
            if parts['restrictions']:
                d['restrictions'] = parse_restrictions(parts['restrictions'])
            return d
        logger.warning('cannot parse package relationship "%s", returning it raw', raw)
        return {'name': raw, 'archqual': None, 'version': None, 'arch': None, 'restrictions': None}
    tl_deps = cls.__comma_sep_RE.split(raw.strip())
    cnf = map(cls.__pipe_sep_RE.split, tl_deps)
    return [[parse_rel(or_dep) for or_dep in or_deps] for or_deps in cnf]