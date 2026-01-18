import textwrap
def check_inline(cmd):
    """Return the inline identifier (may be empty)."""
    cmd._check_compiler()
    body = textwrap.dedent('\n        #ifndef __cplusplus\n        static %(inline)s int static_func (void)\n        {\n            return 0;\n        }\n        %(inline)s int nostatic_func (void)\n        {\n            return 0;\n        }\n        #endif')
    for kw in ['inline', '__inline__', '__inline']:
        st = cmd.try_compile(body % {'inline': kw}, None, None)
        if st:
            return kw
    return ''