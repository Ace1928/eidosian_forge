import dis
import importlib._bootstrap_external
import importlib.machinery
import marshal
import os
import io
import sys
def any_missing_maybe(self):
    """Return two lists, one with modules that are certainly missing
        and one with modules that *may* be missing. The latter names could
        either be submodules *or* just global names in the package.

        The reason it can't always be determined is that it's impossible to
        tell which names are imported when "from module import *" is done
        with an extension module, short of actually importing it.
        """
    missing = []
    maybe = []
    for name in self.badmodules:
        if name in self.excludes:
            continue
        i = name.rfind('.')
        if i < 0:
            missing.append(name)
            continue
        subname = name[i + 1:]
        pkgname = name[:i]
        pkg = self.modules.get(pkgname)
        if pkg is not None:
            if pkgname in self.badmodules[name]:
                missing.append(name)
            elif subname in pkg.globalnames:
                pass
            elif pkg.starimports:
                maybe.append(name)
            else:
                missing.append(name)
        else:
            missing.append(name)
    missing.sort()
    maybe.sort()
    return (missing, maybe)