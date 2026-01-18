import textwrap
def check_gcc_function_attribute_with_intrinsics(cmd, attribute, name, code, include):
    """Return True if the given function attribute is supported with
    intrinsics."""
    cmd._check_compiler()
    body = textwrap.dedent('\n        #include<%s>\n        int %s %s(void)\n        {\n            %s;\n            return 0;\n        }\n\n        int\n        main()\n        {\n            return 0;\n        }\n        ') % (include, attribute, name, code)
    return cmd.try_compile(body, None, None) != 0