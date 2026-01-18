import sys, os
from distutils.errors import \
from distutils.ccompiler import \
from distutils import log
def get_msvc_paths(self, path, platform='x86'):
    """Get a list of devstudio directories (include, lib or path).

        Return a list of strings.  The list will be empty if unable to
        access the registry or appropriate registry keys not found.
        """
    if not _can_read_reg:
        return []
    path = path + ' dirs'
    if self.__version >= 7:
        key = '%s\\%0.1f\\VC\\VC_OBJECTS_PLATFORM_INFO\\Win32\\Directories' % (self.__root, self.__version)
    else:
        key = '%s\\6.0\\Build System\\Components\\Platforms\\Win32 (%s)\\Directories' % (self.__root, platform)
    for base in HKEYS:
        d = read_values(base, key)
        if d:
            if self.__version >= 7:
                return self.__macros.sub(d[path]).split(';')
            else:
                return d[path].split(';')
    if self.__version == 6:
        for base in HKEYS:
            if read_values(base, '%s\\6.0' % self.__root) is not None:
                self.warn('It seems you have Visual Studio 6 installed, but the expected registry settings are not present.\nYou must at least run the Visual Studio GUI once so that these entries are created.')
                break
    return []