import textwrap
def check_compiler_gcc(cmd):
    """Check if the compiler is GCC."""
    cmd._check_compiler()
    body = textwrap.dedent('\n        int\n        main()\n        {\n        #if (! defined __GNUC__)\n        #error gcc required\n        #endif\n            return 0;\n        }\n        ')
    return cmd.try_compile(body, None, None)