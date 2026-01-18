import os
import re
import sys
def compiler_fixup(compiler_so, cc_args):
    """
    This function will strip '-isysroot PATH' and '-arch ARCH' from the
    compile flags if the user has specified one them in extra_compile_flags.

    This is needed because '-arch ARCH' adds another architecture to the
    build, without a way to remove an architecture. Furthermore GCC will
    barf if multiple '-isysroot' arguments are present.
    """
    stripArch = stripSysroot = False
    compiler_so = list(compiler_so)
    if not _supports_universal_builds():
        stripArch = stripSysroot = True
    else:
        stripArch = '-arch' in cc_args
        stripSysroot = any((arg for arg in cc_args if arg.startswith('-isysroot')))
    if stripArch or 'ARCHFLAGS' in os.environ:
        while True:
            try:
                index = compiler_so.index('-arch')
                del compiler_so[index:index + 2]
            except ValueError:
                break
    elif not _supports_arm64_builds():
        for idx in reversed(range(len(compiler_so))):
            if compiler_so[idx] == '-arch' and compiler_so[idx + 1] == 'arm64':
                del compiler_so[idx:idx + 2]
    if 'ARCHFLAGS' in os.environ and (not stripArch):
        compiler_so = compiler_so + os.environ['ARCHFLAGS'].split()
    if stripSysroot:
        while True:
            indices = [i for i, x in enumerate(compiler_so) if x.startswith('-isysroot')]
            if not indices:
                break
            index = indices[0]
            if compiler_so[index] == '-isysroot':
                del compiler_so[index:index + 2]
            else:
                del compiler_so[index:index + 1]
    sysroot = None
    argvar = cc_args
    indices = [i for i, x in enumerate(cc_args) if x.startswith('-isysroot')]
    if not indices:
        argvar = compiler_so
        indices = [i for i, x in enumerate(compiler_so) if x.startswith('-isysroot')]
    for idx in indices:
        if argvar[idx] == '-isysroot':
            sysroot = argvar[idx + 1]
            break
        else:
            sysroot = argvar[idx][len('-isysroot'):]
            break
    if sysroot and (not os.path.isdir(sysroot)):
        sys.stderr.write(f"Compiling with an SDK that doesn't seem to exist: {sysroot}\n")
        sys.stderr.write('Please check your Xcode installation\n')
        sys.stderr.flush()
    return compiler_so