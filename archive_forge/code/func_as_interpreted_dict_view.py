import collections.abc
import contextlib
import sys
import textwrap
import weakref
from abc import ABC
from types import TracebackType
from weakref import ReferenceType
from debian._deb822_repro._util import (combine_into_replacement, BufferingIterator,
from debian._deb822_repro.formatter import (
from debian._deb822_repro.tokens import (
from debian._deb822_repro.types import AmbiguousDeb822FieldKeyError, SyntaxOrParseError
from debian._util import (
def as_interpreted_dict_view(self, interpretation, *, auto_resolve_ambiguous_fields=True):
    """Provide a Dict-like view of the paragraph

        This method returns a dict-like object representing this paragraph and
        is useful for accessing fields in a given interpretation. It is possible
        to use multiple versions of this dict-like view with different interpretations
        on the same paragraph at the same time (for different fields).

            >>> example_deb822_paragraph = '''
            ... Package: foo
            ... # Field comment (because it becomes just before a field)
            ... Architecture: amd64
            ... # Inline comment (associated with the next line)
            ...               i386
            ... # We also support arm
            ...               arm64
            ...               armel
            ... '''
            >>> dfile = parse_deb822_file(example_deb822_paragraph.splitlines())
            >>> paragraph = next(iter(dfile))
            >>> list_view = paragraph.as_interpreted_dict_view(LIST_SPACE_SEPARATED_INTERPRETATION)
            >>> # With the defaults, you only deal with the semantic values
            >>> # - no leading or trailing whitespace on the first part of the value
            >>> list(list_view["Package"])
            ['foo']
            >>> with list_view["Architecture"] as arch_list:
            ...     orig_arch_list = list(arch_list)
            ...     arch_list.replace('i386', 'kfreebsd-amd64')
            >>> orig_arch_list
            ['amd64', 'i386', 'arm64', 'armel']
            >>> list(list_view["Architecture"])
            ['amd64', 'kfreebsd-amd64', 'arm64', 'armel']
            >>> print(paragraph.dump(), end='')
            Package: foo
            # Field comment (because it becomes just before a field)
            Architecture: amd64
            # Inline comment (associated with the next line)
                          kfreebsd-amd64
            # We also support arm
                          arm64
                          armel
            >>> # Format preserved and architecture replaced
            >>> with list_view["Architecture"] as arch_list:
            ...     # Prettify the result as sorting will cause awkward whitespace
            ...     arch_list.reformat_when_finished()
            ...     arch_list.sort()
            >>> print(paragraph.dump(), end='')
            Package: foo
            # Field comment (because it becomes just before a field)
            Architecture: amd64
            # We also support arm
                          arm64
                          armel
            # Inline comment (associated with the next line)
                          kfreebsd-amd64
            >>> list(list_view["Architecture"])
            ['amd64', 'arm64', 'armel', 'kfreebsd-amd64']
            >>> # Format preserved and architecture values sorted

        :param interpretation: Decides how the field values are interpreted.  As an example,
          use LIST_SPACE_SEPARATED_INTERPRETATION for fields such as Architecture in the
          debian/control file.
        :param auto_resolve_ambiguous_fields: This parameter is only relevant for paragraphs
          that contain the same field multiple times (these are generally invalid).  If the
          caller requests an ambiguous field from an invalid paragraph via a plain field name,
          the return dict-like object will refuse to resolve the field (not knowing which
          version to pick).  This parameter (if set to True) instead changes the error into
          assuming the caller wants the *first* variant.
        """
    return Deb822InterpretingParagraphWrapper(self, interpretation, auto_resolve_ambiguous_fields=auto_resolve_ambiguous_fields)