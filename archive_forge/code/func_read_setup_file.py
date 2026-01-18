import os
import re
import warnings
def read_setup_file(filename):
    """Reads a Setup file and returns Extension instances."""
    from distutils.sysconfig import parse_makefile, expand_makefile_vars, _variable_rx
    from distutils.text_file import TextFile
    from distutils.util import split_quoted
    vars = parse_makefile(filename)
    file = TextFile(filename, strip_comments=1, skip_blanks=1, join_lines=1, lstrip_ws=1, rstrip_ws=1)
    try:
        extensions = []
        while True:
            line = file.readline()
            if line is None:
                break
            if re.match(_variable_rx, line):
                continue
            if line[0] == line[-1] == '*':
                file.warn("'%s' lines not handled yet" % line)
                continue
            line = expand_makefile_vars(line, vars)
            words = split_quoted(line)
            module = words[0]
            ext = Extension(module, [])
            append_next_word = None
            for word in words[1:]:
                if append_next_word is not None:
                    append_next_word.append(word)
                    append_next_word = None
                    continue
                suffix = os.path.splitext(word)[1]
                switch = word[0:2]
                value = word[2:]
                if suffix in ('.c', '.cc', '.cpp', '.cxx', '.c++', '.m', '.mm'):
                    ext.sources.append(word)
                elif switch == '-I':
                    ext.include_dirs.append(value)
                elif switch == '-D':
                    equals = value.find('=')
                    if equals == -1:
                        ext.define_macros.append((value, None))
                    else:
                        ext.define_macros.append((value[0:equals], value[equals + 2:]))
                elif switch == '-U':
                    ext.undef_macros.append(value)
                elif switch == '-C':
                    ext.extra_compile_args.append(word)
                elif switch == '-l':
                    ext.libraries.append(value)
                elif switch == '-L':
                    ext.library_dirs.append(value)
                elif switch == '-R':
                    ext.runtime_library_dirs.append(value)
                elif word == '-rpath':
                    append_next_word = ext.runtime_library_dirs
                elif word == '-Xlinker':
                    append_next_word = ext.extra_link_args
                elif word == '-Xcompiler':
                    append_next_word = ext.extra_compile_args
                elif switch == '-u':
                    ext.extra_link_args.append(word)
                    if not value:
                        append_next_word = ext.extra_link_args
                elif suffix in ('.a', '.so', '.sl', '.o', '.dylib'):
                    ext.extra_objects.append(word)
                else:
                    file.warn("unrecognized argument '%s'" % word)
            extensions.append(ext)
    finally:
        file.close()
    return extensions