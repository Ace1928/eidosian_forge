import textwrap
def check_gcc_version_at_least(cmd, major, minor=0, patchlevel=0):
    """
    Check that the gcc version is at least the specified version."""
    cmd._check_compiler()
    version = '.'.join([str(major), str(minor), str(patchlevel)])
    body = textwrap.dedent('\n        int\n        main()\n        {\n        #if (! defined __GNUC__) || (__GNUC__ < %(major)d) || \\\n                (__GNUC_MINOR__ < %(minor)d) || \\\n                (__GNUC_PATCHLEVEL__ < %(patchlevel)d)\n        #error gcc >= %(version)s required\n        #endif\n            return 0;\n        }\n        ')
    kw = {'version': version, 'major': major, 'minor': minor, 'patchlevel': patchlevel}
    return cmd.try_compile(body % kw, None, None)