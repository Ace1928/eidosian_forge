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
def parse_deb822_file(sequence, *, accept_files_with_error_tokens=False, accept_files_with_duplicated_fields=False, encoding='utf-8'):
    """

    :param sequence: An iterable over lines of str or bytes (an open file for
      reading will do).  If line endings are provided in the input, then they
      must be present on every line (except the last) will be preserved as-is.
      If omitted and the content is at least 2 lines, then parser will assume
      implicit newlines.
    :param accept_files_with_error_tokens: If True, files with critical syntax
      or parse errors will be returned as "successfully" parsed. Usually,
      working on files with this kind of errors are not desirable as it is
      hard to make sense of such files (and they might in fact not be a deb822
      file at all).  When set to False (the default) a ValueError is raised if
      there is a critical syntax or parse error.
      Note that duplicated fields in a paragraph is not considered a critical
      parse error by this parser as the implementation can gracefully cope
      with these. Use accept_files_with_duplicated_fields to determine if
      such files should be accepted.
    :param accept_files_with_duplicated_fields: If True, then
      files containing paragraphs with duplicated fields will be returned as
      "successfully" parsed even though they are invalid according to the
      specification.  The paragraphs will prefer the first appearance of the
      field unless caller explicitly requests otherwise (e.g., via
      Deb822ParagraphElement.configured_view).  If False, then this method
      will raise a ValueError if any duplicated fields are seen inside any
      paragraph.
    :param encoding: The encoding to use (this is here to support Deb822-like
       APIs, new code should not use this parameter).
    """
    if isinstance(sequence, (str, bytes)):
        sequence = sequence.splitlines(True)
    tokens = tokenize_deb822_file(sequence, encoding=encoding)
    if not accept_files_with_error_tokens:
        tokens = _abort_on_error_tokens(tokens)
    tokens = _combine_comment_tokens_into_elements(tokens)
    tokens = _build_value_line(tokens)
    tokens = _combine_vl_elements_into_value_elements(tokens)
    tokens = _build_field_with_value(tokens)
    tokens = _combine_kvp_elements_into_paragraphs(tokens)
    tokens = _combine_error_tokens_into_elements(tokens)
    deb822_file = Deb822FileElement(LinkedList(tokens))
    if not accept_files_with_duplicated_fields:
        for no, paragraph in enumerate(deb822_file):
            if isinstance(paragraph, Deb822DuplicateFieldsParagraphElement):
                field_names = set()
                dup_field = None
                for field in paragraph.keys():
                    field_name, _, _ = _unpack_key(field)
                    assert isinstance(field_name, str)
                    if field_name in field_names:
                        dup_field = field_name
                        break
                    field_names.add(field_name)
                if dup_field is not None:
                    msg = 'Duplicate field "{dup_field}" in paragraph number {no}'
                    raise ValueError(msg.format(dup_field=dup_field, no=no))
    return deb822_file