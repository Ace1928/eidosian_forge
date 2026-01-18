import logging
from cmakelang.lex import TokenType
from cmakelang.parse.additional_nodes import PatternNode
from cmakelang.parse.argument_nodes import (
from cmakelang.parse.util import get_first_semantic_token
from cmakelang.parse.simple_nodes import consume_whitespace_and_comments
def parse_file(ctx, tokens, breakstack):
    """
  The ``file()`` command has a lot of different forms, depending on the first
  argument. This function just dispatches the correct parse implementation for
  the given form::

    Reading
      file(READ <filename> <out-var> [...])
      file(STRINGS <filename> <out-var> [...])
      file(<HASH> <filename> <out-var>)
      file(TIMESTAMP <filename> <out-var> [...])

    Writing
      file({WRITE | APPEND} <filename> <content>...)
      file({TOUCH | TOUCH_NOCREATE} [<file>...])
      file(GENERATE OUTPUT <output-file> [...])

    Filesystem
      file({GLOB | GLOB_RECURSE} <out-var> [...] [<globbing-expr>...])
      file(RENAME <oldname> <newname>)
      file({REMOVE | REMOVE_RECURSE } [<files>...])
      file(MAKE_DIRECTORY [<dir>...])
      file({COPY | INSTALL} <file>... DESTINATION <dir> [...])
      file(SIZE <filename> <out-var>)
      file(READ_SYMLINK <linkname> <out-var>)
      file(CREATE_LINK <original> <linkname> [...])

    Path Conversion
      file(RELATIVE_PATH <out-var> <directory> <file>)
      file({TO_CMAKE_PATH | TO_NATIVE_PATH} <path> <out-var>)

    Transfer
      file(DOWNLOAD <url> <file> [...])
      file(UPLOAD <file> <url> [...])

    Locking
      file(LOCK <path> [...])

  :see: https://cmake.org/cmake/help/v3.14/command/file.html
  """
    descriminator_token = get_first_semantic_token(tokens)
    if descriminator_token is None or descriminator_token.type is TokenType.RIGHT_PAREN:
        location = ()
        if tokens:
            location = tokens[0].get_location()
        ctx.lint_ctx.record_lint('E1120', location=location)
        return StandardArgTree.parse(ctx, tokens, npargs='*', kwargs={}, flags=[], breakstack=breakstack)
    if descriminator_token.type is TokenType.DEREF:
        ctx.lint_ctx.record_lint('C0114', location=descriminator_token.get_location())
        return StandardArgTree.parse(ctx, tokens, npargs='*', kwargs={}, flags=[], breakstack=breakstack)
    descriminator = descriminator_token.spelling.upper()
    parsemap = {'READ': parse_file_read, 'STRINGS': parse_file_strings, 'TIMESTAMP': parse_file_timestamp, 'WRITE': parse_file_write, 'APPEND': parse_file_write, 'TOUCH': StandardParser('+', flags=['TOUCH']), 'TOUCH_NOCREATE': StandardParser('+', flags=['TOUCH_NOCREATE']), 'GENERATE': parse_file_generate_output, 'GLOB': parse_file_glob, 'GLOB_RECURSE': parse_file_glob, 'RENAME': StandardParser(3, flags=['RENAME']), 'REMOVE': StandardParser('+', flags=['REMOVE']), 'REMOVE_RECURSE': StandardParser('+', flags=['REMOVE_RECURSE']), 'MAKE_DIRECTORY': StandardParser('+', flags=['MAKE_DIRECTORY']), 'COPY': parse_file_copy, 'INSTALL': parse_file_copy, 'SIZE': StandardParser(3, flags=['SIZE']), 'READ_SYMLINK': StandardParser(3, flags=['READ_SYMLINK']), 'CREATE_LINK': parse_file_create_link, 'RELATIVE_PATH': StandardParser(4, flags=['RELATIVE_PATH']), 'TO_CMAKE_PATH': StandardParser(3, flags=['TO_CMAKE_PATH']), 'TO_NATIVE_PATH': StandardParser(3, flags=['TO_NATIVE_PATH']), 'DOWNLOAD': parse_file_xfer, 'UPLOAD': parse_file_xfer, 'LOCK': parse_file_lock}
    for hashname in HASH_STRINGS:
        parsemap[hashname] = parse_file_hash
    if descriminator not in parsemap:
        ctx.lint_ctx.record_lint('E1126', location=descriminator_token.get_location())
        return StandardArgTree.parse(ctx, tokens, npargs='*', kwargs={}, flags=[], breakstack=breakstack)
    return parsemap[descriminator](ctx, tokens, breakstack)